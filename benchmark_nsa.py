import torch
import triton
from flash_attn import flash_attn_func

from nsa import nsa_func

from fla.ops.nsa import parallel_nsa as nsa_fla

from native_sparse_attention.ops import linear_compress, compressed_attention, topk_sparse_attention

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['L'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['nsa', 'nsa_bwd', 'nsa_fla', 'nsa_fla_bwd', 'nsa_tilelang', 'nsa_tilelang_bwd', 'nsa_xunhaolai', 'nsa_xunhaolai_bwd', 'nsa_lucidrains', 'nsa_lucidrains_bwd', 'nsa_falai', 'nsa_falai_bwd', 'flash', 'flash_bwd'],
        # label name for the lines
        line_names=['nsa', 'nsa_bwd', 'nsa_fla', 'nsa_fla_bwd', 'nsa_tilelang', 'nsa_tilelang_bwd', 'nsa_xunhaolai', 'nsa_xunhaolai_bwd', 'nsa_lucidrains', 'nsa_lucidrains_bwd', 'nsa_falai', 'nsa_falai_bwd', 'flash', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'),
                ('blue', 'dotted'), ('red', 'dotted'), ('cyan', '-'), ('cyan', 'dotted')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(L, provider):
    device = "cuda"
    dtype = torch.bfloat16
    requires_grad = True
    B, HQ, HK, DK, DV = 2, 16, 1, 16, 16
    block_size = 64
    block_stride = 64
    block_selections = 8
    swa_size = 512

    q = torch.randn(B, L, HQ, DK, requires_grad=requires_grad, device=device)
    if 'nsa' in provider:
        if provider in ['nsa', 'nsa_bwd']:
            k = torch.randn(B, L, 3*HK, DK, requires_grad=requires_grad, device=device)
            v = torch.randn(B, L, 3*HK, DV, requires_grad=requires_grad, device=device)
            w_k = torch.randn(HK, block_size*DK, DK, requires_grad=requires_grad, device=device)
            w_v = torch.randn(HK, block_size*DV, DV, requires_grad=requires_grad, device=device)
            pe_k = torch.randn(HK, block_size, DK, requires_grad=requires_grad, device=device)
            pe_v = torch.randn(HK, block_size, DV, requires_grad=requires_grad, device=device)
        else:
            k = torch.randn(B, L, HK, DK, requires_grad=requires_grad, device=device)
            v = torch.randn(B, L, HK, DV, requires_grad=requires_grad, device=device)
        g_cmp = torch.randn(B, L, HQ, requires_grad=requires_grad, device=device)
        g_slc = torch.randn(B, L, HQ, requires_grad=requires_grad, device=device)
        g_swa = torch.randn(B, L, HQ, requires_grad=requires_grad, device=device)
    else:
        k = torch.randn(B, L, HK, DK, requires_grad=requires_grad, device=device)
        v = torch.randn(B, L, HK, DV, requires_grad=requires_grad, device=device)
    do = torch.ones(B, L, HQ, DV, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    match provider:
        case 'nsa':
            results = triton.testing.do_bench(lambda: nsa_func(q, k, v, g_cmp, g_slc, g_swa, w_k, w_v, pe_k, pe_v, block_size, block_stride, swa_size, block_selections), quantiles=quantiles)
        case 'nsa_bwd':
            results = triton.testing.do_bench(lambda: nsa_func(q, k, v, g_cmp, g_slc, g_swa, w_k, w_v, pe_k, pe_v, block_size, block_stride, swa_size, block_selections).backward(do), quantiles=quantiles)
        case 'nsa_fla':
            results = triton.testing.do_bench(lambda: nsa_fla(q, k, v, g_cmp, g_slc, g_swa, block_selections, block_size, swa_size), quantiles=quantiles)
        case 'nsa_fla_bwd':
            results = triton.testing.do_bench(lambda: nsa_fla(q, k, v, g_cmp, g_slc, g_swa, block_selections, block_size, swa_size).backward(do), quantiles=quantiles)
        case 'flash':
            results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True), quantiles=quantiles)
        case 'flash_bwd':
            results = triton.testing.do_bench(lambda: flash_attn_func(q, k, v, causal=True).backward(do), quantiles=quantiles)
        case _:
            results = 0, 0, 0

    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')
