# GPU KV fraction vs inference speed

LLaMa2-70B / 1×H200 / paged-attn / burst 100 requests (prefill & decode 512±32).  
`gpu_frac` caps GPU KV occupancy: each request charges `floor(S × gpu_frac)` HBM tokens. Off-GPU remainder is DRAM:SSD = 5:2 with page-fault fetch. SSD is N3X-SLC 4K / `qd_cap=32` / 14 GB/s (same as official Case 2).

Source: [`kv_working_set_test_report/gpu_frac_sweep/summary.json`](kv_working_set_test_report/gpu_frac_sweep/summary.json)

**Peak B** (decode, S=1024) is the HBM concurrency ceiling after watermark:

```text
Peak B = 513 / ceil(floor(1024 × gpu_frac) / 16)
```

513 = 518 leftover GPU blocks − 5 watermark. This is a **block-rounded** integer, so several `gpu_frac` values share one B — that is the staircase.

---

## 1. System token/s vs GPU KV fraction — peak B on the same axes

![System token/s vs gpu_frac with peak-B plateaus](kv_working_set_test_report/gpu_frac_sweep/fig_tokens_peak_b.png)

How to read this figure / 怎么读这张图:

- **Left axis, blue:** system token/s = (prefill + decode tokens) / makespan.
- **Right axis, orange step:** peak B at S=1024. A step stays flat until GPU block rounding admits one more concurrent request.
- **Shaded bands:** one band = one constant peak B. Token/s is mostly flat **inside** a band and jumps when the orange step jumps.

所以吞吐台阶不是 5% 本身有意义，而是 **peak B 变了**。`gpu_frac` 100%→95% 同为 B=8，token/s 几乎不动（292.0 → 292.7）；B 从 8 升到 9（约 85%→80%）才跳到 352 tok/s。

Exception inside the B=8 band / B=8 色带里的例外: at 90% token/s rises to 321 while **decode** peak B is still 8. Prefill peak B (S=512) goes 16→17, so the first packed prefill admits one extra request (see §4). Decode B 仍是主阶梯；prefill B 会在同一 decode-B 色带里造成一次小跳。

| gpu_frac | Peak B | token/s | vs 100% |
| ---: | ---: | ---: | ---: |
| 100% | 8 | 292 | 1.00× |
| 50% | 16 | 505 | 1.73× |
| 30% | 25 | 711 | 2.44× |
| 10% | 73 | 1077 | 3.69× |

---

## 2. Direct view: token/s as a function of peak B

![System token/s vs peak B](kv_working_set_test_report/gpu_frac_sweep/fig_tokens_vs_peak_b.png)

横轴换成 peak B 以后，吞吐沿一条上升曲线走，和 `gpu_frac` 的 5% 刻度无关。关系是 **次线性**的：B 从 8 到 73（9.1×），token/s 只到 3.69×，因为更大的 decode batch 抬高了 TPOT（attention 按请求求和）。

---

## 3. TTFT vs TPOT (same sweep)

![TTFT p50 and TPOT p50 vs gpu_frac](kv_working_set_test_report/gpu_frac_sweep/fig_ttft_tpot_frac.png)

Peak B 变大以后：

- **TTFT p50 下降** — 更多请求能一起做 packed prefill，排队变短（100% 时 p50≈142 s，30% 起第一波就能进大部分 burst）。
- **TPOT p50 上升** — 同一步 decode 绑着更多请求，ITL 变长（69 ms → 142 ms）。

系统更快、单流更慢，是同一并发上升的两面。

---

## 4. Prefill peak B vs decode peak B

![Prefill vs decode peak B](kv_working_set_test_report/gpu_frac_sweep/fig_peak_b_prefill_decode.png)

S=512 时每条请求占的 GPU block 更少，所以 **B_prefill > B_decode**。90% 那一次 token/s 小跳，对应蓝线 16→17，橙线仍停在 8。主图上的大台阶仍跟橙线（满长 decode B）对齐。

---

## 5. Full sweep table

| gpu_frac | Peak B (S=1024) | B_pre (S=512) | token/s | TPOT p50 | TTFT p50 | makespan | preempt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 16 | 292.0 | 69.0 ms | 141.9 s | 350.9 s | 73 |
| 95% | 8 | 16 | 292.7 | 69.8 ms | 142.3 s | 350.0 s | 72 |
| 90% | 8 | 17 | 320.9 | 70.2 ms | 141.9 s | 319.3 s | 64 |
| 85% | 9 | 18 | 321.4 | 70.9 ms | 110.1 s | 318.7 s | 65 |
| 80% | 9 | 19 | 351.8 | 71.9 ms | 111.3 s | 291.2 s | 65 |
| 75% | 10 | 21 | 353.4 | 72.8 ms | 112.2 s | 289.9 s | 66 |
| 70% | 11 | 22 | 390.6 | 74.1 ms | 112.6 s | 262.3 s | 63 |
| 65% | 12 | 24 | 391.7 | 75.1 ms | 79.7 s | 261.5 s | 65 |
| 60% | 13 | 25 | 439.5 | 76.6 ms | 80.4 s | 233.1 s | 64 |
| 55% | 14 | 28 | 443.0 | 78.7 ms | 81.0 s | 231.3 s | 64 |
| 50% | 16 | 32 | 505.4 | 80.5 ms | 81.8 s | 202.7 s | 57 |
| 45% | 17 | 34 | 505.5 | 82.7 ms | 77.9 s | 202.7 s | 63 |
| 40% | 19 | 39 | 591.9 | 85.5 ms | 47.4 s | 173.1 s | 54 |
| 35% | 22 | 42 | 592.8 | 89.1 ms | 48.5 s | 172.8 s | 57 |
| 30% | 25 | 51 | 711.1 | 92.6 ms | 2.94 s | 144.1 s | 46 |
| 25% | 32 | 64 | 717.3 | 98.7 ms | 3.51 s | 142.8 s | 50 |
| 20% | 39 | 73 | 835.2 | 108.4 ms | 4.20 s | 122.7 s | 41 |
| 15% | 51 | 102 | 908.2 | 126.1 ms | 5.77 s | 112.8 s | 49 |
| 10% | 73 | 128 | 1076.8 | 141.8 ms | 5.77 s | 95.1 s | 26 |

DRAM/SSD capacity is infinite in this model. Implied peak occupancy at S=1024 (not enforced): **30% → 31.3 GiB DRAM + 12.5 GiB SSD**; **10% → 117.3 GiB DRAM + 47.1 GiB SSD**. Full table in [`kv_working_set_test_report.md`](kv_working_set_test_report.md) §8. Σ fetch at 10% is 7.40 s vs 95 s makespan.

Regenerate figures:

```bash
python3.11 kv_working_set_test_report/gpu_frac_sweep/plot_gpu_frac.py
```
