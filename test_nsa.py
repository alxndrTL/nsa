import torch
torch.set_float32_matmul_precision('high')

from nsa import nsa_func

import time

B, L, HQ, HK, DK, DV = 2, 4096, 16, 1, 16, 16
block_size = 64
block_stride = 64
block_selections = 8
swa_size = 512

torch.cuda.reset_peak_memory_stats()

q = torch.randn(B, L, HQ, DK).to("cuda")
k = torch.randn(B, L, 3*HK, DK).to("cuda")
v = torch.randn(B, L, 3*HK, DV).to("cuda")
g_cmp = torch.randn(B, L, HQ).to("cuda")
g_slc = torch.randn(B, L, HQ).to("cuda")
g_swa = torch.randn(B, L, HQ).to("cuda")
w_k = torch.randn(HK, block_size*DK, DK).to("cuda")
w_v = torch.randn(HK, block_size*DV, DV).to("cuda")
pe_k = torch.randn(HK, block_size, DK).to("cuda")
pe_v = torch.randn(HK, block_size, DV).to("cuda")

flash_nsa_func = torch.compile(nsa_func)

# warmup
for _ in range(5):
    flash_nsa_func(q, k, v, g_cmp, g_slc, g_swa, w_k, w_v, pe_k, pe_v, block_size, block_stride, swa_size, block_selections)

N=5
st = time.time()
for _ in range(N):
    o = flash_nsa_func(q, k, v, g_cmp, g_slc, g_swa, w_k, w_v, pe_k, pe_v, block_size, block_stride, swa_size, block_selections)
print(f"total time: {(time.time() - st)*1000/N:.2f} ms")
print(f"total memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} Mo")
