import torch

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import time

B, L, HQ, HK, DK, DV = 2, 4096, 16, 1, 16, 16

torch.cuda.reset_peak_memory_stats()

q = torch.randn(B, L, HQ, DK).to("cuda")
k = torch.randn(B, L, HK, DK).to("cuda")
v = torch.randn(B, L, HK, DV).to("cuda")

flex_attention = torch.compile(flex_attention)

def causal_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
block_mask = create_block_mask(causal_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, _compile=True)

# warmup
for _ in range(5):
    flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True).transpose(1, 2)

N=5
st = time.time()
for _ in range(N):
    o = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True).transpose(1, 2)
print(f"total time: {(time.time() - st)*1000/N:.2f} ms")
print(f"total memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} Mo")
