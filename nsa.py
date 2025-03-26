import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from fla.ops.nsa.parallel import parallel_nsa_topk as select_topk

#note: flex_attention forces DK and DV to be powers of 2

def nsa_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cmp: torch.Tensor, g_slc: torch.Tensor, g_swa: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, pe_k: torch.Tensor, pe_v: torch.Tensor, block_size: int, block_stride: int, swa_size: int, n_selections: int):
    # q: (B, L, H, DK)
    # k: (B, L, 3H, DK)
    # v: (B, L, 3H, DV)
    # g_cmp, g_slc, g_swa: (B, L, H)
    # w_k, w_v: (H, block_size*DK, D)
    # pe_k, pe_v: (H, block_size, D)
    
    # output: (B, L, H, DV)

    B, L, HQ, DK = q.size()
    HK = k.size(2)//3
    DV = v.size(-1)

    assert DK==DV, "for now"

    k_swa, k_cmp, k_slc = torch.chunk(k, 3, dim=2)
    v_swa, v_cmp, v_slc = torch.chunk(v, 3, dim=2)

    # todo: stride
    # todo: different block size between compression and selection

    # compute swa
    # regular attention, just limited to a short fixed window (something like 512)

    # todo: generate this mask outsite of the function, as it's fixed
    def causal_swa_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= swa_size
        return causal_mask & window_mask
    block_mask = create_block_mask(causal_swa_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True)

    o_swa = flex_attention(q.transpose(1, 2), k_swa.transpose(1, 2), v_swa.transpose(1, 2), block_mask=block_mask, enable_gqa=True).transpose(1, 2) # (B, Lq, H, DV)
    o = g_swa.unsqueeze(-1) * o_swa

    # compute compression
    # average the k,v across the seqlen into blocks (L//block_size k,v instead of L). do the attention between the L queries and the L//block_size averaged k,v
    # this acts as a corse grained attention

    def causal_block_mask(b, h, q_idx, kv_block_idx):
        return q_idx >= ((kv_block_idx + 1) * block_size - 1)
    block_mask = create_block_mask(causal_block_mask, B=None, H=None, Q_LEN=L, KV_LEN=L//block_size)

    # linear compress
    assert block_size==block_stride, "for now"
    #k_cmp = k_cmp.view(B, L//block_size, block_size, HK, DK).mean(dim=2) # (B, L//block_size, HK, DK)
    #v_cmp = v_cmp.view(B, L//block_size, block_size, HK, DV).mean(dim=2) # (B, L//block_size, HK, DV)
    k_cmp = linear_compress(k_cmp, w_k, pe=pe_k, block_size=block_size, block_stride=block_stride) # (B, L//block_size, HK, DK)
    v_cmp = linear_compress(v_cmp, w_v, pe=pe_v, block_size=block_size, block_stride=block_stride) # (B, L//block_size, HK, DV)

    o_cmp, lse_cmp = flex_attention(q.transpose(1, 2), k_cmp.transpose(1, 2), v_cmp.transpose(1, 2), block_mask=block_mask, enable_gqa=True, return_lse=True) # (B, H, Lq, DV), (B, H, Lq)
    o_cmp = o_cmp.transpose(1, 2) # (B, L, H, DV)
    lse_cmp = lse_cmp.transpose(1, 2) # (B, L, H)
    o += g_cmp.unsqueeze(-1) * o_cmp

    # the first block_size-1 queries don't attend to anything in this attn (but that's ok)
    
    # compute selection
    # here, we compute regular attention, with the exception that we mask out blocks of k,v (and hence dont compute attn over these)
    # selection is done with a top-n selection + keep the first block no matter what

    block_indices = select_topk(q, k_cmp, lse=lse_cmp, block_counts=n_selections, block_size=block_size, scale=DK**-0.5).transpose(1, 2) # (B, HK, Lq, n_select)
    block_indices = block_indices.repeat_interleave(HQ//HK, dim=1) # (B, HQ, Lq, n_select)
    block_indices_boolean = torch.arange(L//block_size, device=block_indices.device)
    select_boolean = (block_indices.unsqueeze(-1) == block_indices_boolean).any(dim=-2).bool()

    def selection_mod(b, h, q_idx, kv_idx):
        block_idx = kv_idx // block_size

        first_block_mask = block_idx == 0
        prev_block_mask = (block_idx == (q_idx//block_size - 1))
        pprev_block_mask = (block_idx == (q_idx//block_size - 2))
        causal_mask = q_idx >= kv_idx
        selection_mask = select_boolean[b, h, q_idx, block_idx]
        return causal_mask & (selection_mask | first_block_mask | prev_block_mask | pprev_block_mask)
    block_mask = create_block_mask(selection_mod, B=B, H=HQ, Q_LEN=L, KV_LEN=L, _compile=True)
    o_slc = flex_attention(q.transpose(1, 2), k_slc.transpose(1, 2), v_slc.transpose(1, 2), block_mask=block_mask, enable_gqa=True).transpose(1, 2) # (B, L, HQ, DV)
    o += g_slc.unsqueeze(-1) * o_slc # (B, L, H, DV)

    return o

# adapted from github.com/XunhaoLai/native-sparse-attention-triton
def linear_compress(x: torch.Tensor, w: torch.Tensor, pe: torch.Tensor, block_size: int, block_stride: int):
    """
    x: (B, L, H, D) (either k or v)
    w: (H, block_size*D, D) (weight)
    pe: (H, block_size, D)

    output: (B, L_compressed, H, D)
    """

    B, L, H, D = x.shape
    L_compressed = (L - block_size) // block_stride + 1

    # Extract blocks with shape: (B, L_compressed, H, block_size, D)
    x_blocks = x.unfold(dimension=1, size=block_size, step=block_stride)
    # Flatten each block to shape: (B, L_compressed, H, block_size*D)
    x_blocks = x_blocks.reshape(B, L_compressed, H, block_size * D)

    # Apply the learned linear transformation per head:
    # For each head h, compute: out[b, n, h, :] = x_blocks[b, n, h, :] @ w[h]
    y = torch.einsum("bnhd, hdf -> bnhf", x_blocks, w)

    # Compute and add positional bias:
    # Flatten pe from (H, block_size, D) to (H, block_size*D) and project it with w.
    if pe is not None:
        pe_flat = pe.reshape(H, block_size * D)
        bias = torch.einsum("hD, hDd -> hd", pe_flat, w)  # shape: (H, D)
        y = y + bias.unsqueeze(0).unsqueeze(0)
    return y
