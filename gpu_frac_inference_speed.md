# GPU KV fraction vs inference speed

LLaMa2-70B / 1×H200 / paged-attn / burst 100 requests (prefill & decode 512±32).
Prefill / recompute charge **full** context on HBM. After those steps, decode occupancy is `floor(S × gpu_frac)` HBM tokens. Off-GPU remainder is DRAM:SSD = 5:2 and is read **every decode step**. SSD is N3X-SLC 4K / `qd_cap=32` / 14 GB/s (same as official Case 2).

Source: [`kv_working_set_test_report/gpu_frac_sweep/summary.json`](kv_working_set_test_report/gpu_frac_sweep/summary.json)

**Peak B** (decode, S=1024) is the HBM concurrency ceiling after watermark:

```text
Peak B = 513 / ceil(floor(1024 × gpu_frac) / 16)
```

513 = 518 leftover GPU blocks − 5 watermark. Prefill Peak B at S=512 is always **16** (full prompt on GPU).

---

## 1. System token/s vs GPU KV fraction — peak B on the same axes

![System token/s vs gpu_frac with peak-B plateaus](kv_working_set_test_report/gpu_frac_sweep/fig_tokens_peak_b.png)

How to read this figure / 怎么读这张图:

- **Left axis, blue:** system token/s = (prefill + decode tokens) / makespan.
- **Right axis, orange step:** peak B at S=1024. Lower `gpu_frac` admits more concurrent decode requests **and** a larger per-step cold set.
- Token/s **falls** as `gpu_frac` falls: every decode token re-reads `[0, gpu_start)`, so I/O dominates. Peak B going up does not raise throughput here.

吞吐随 `gpu_frac` 下降：冷 KV 每步都读，I/O 盖过占用带来的并发。100%→30% token/s 从 292 降到 32.5（0.11×）。

| gpu_frac | Peak B | token/s | vs 100% |
| ---: | ---: | ---: | ---: |
| 100% | 8 | 292 | 1.00× |
| 50% | 16 | 43.5 | 0.15× |
| 30% | 25 | 32.5 | 0.11× |
| 10% | 73 | 25.9 | 0.09× |

---

## 2. Direct view: token/s as a function of peak B

![System token/s vs peak B](kv_working_set_test_report/gpu_frac_sweep/fig_tokens_vs_peak_b.png)

横轴换成 peak B 以后，吞吐沿一条**下降**曲线走：B 从 8 到 73 时 token/s 从 292 降到 26。更大的 offload 窗口等于每步更多 SSD/DRAM 流量，共享队列上的 batch 越大越慢。

---

## 3. TTFT vs TPOT (same sweep)

![TTFT p50 and TPOT p50 vs gpu_frac](kv_working_set_test_report/gpu_frac_sweep/fig_ttft_tpot_frac.png)

- **TPOT p50 单调上升** — 冷集变大，每步 I/O 变长（69 ms → 7.0 s）。
- **TTFT p50 先升后掉** — 30% 及以上，后续 prompt 排在慢 decode 后面（p50 到 1091 s）。25% 及以下，裁块后腾出的 HBM 够把剩余 burst 很快 prefill 完，p50 回到 ~3 s；makespan 仍然更长。

系统更慢、单流更慢。TTFT 在低 `gpu_frac` 变好只是因为 prefill 提前结束，不是 decode 变快。

---

## 4. Prefill peak B vs decode peak B

![Prefill vs decode peak B](kv_working_set_test_report/gpu_frac_sweep/fig_peak_b_prefill_decode.png)

Prefill 占位按全量上下文，S=512 时 **B_prefill = 16** 不随 `gpu_frac` 变。Decode 裁到 `gpu_frac` 后 B_decode 从 8 升到 73。

---

## 5. Full sweep table

| gpu_frac | Peak B (S=1024) | B_pre (S=512) | token/s | TPOT p50 | TTFT p50 | makespan | preempt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 16 | 292.0 | 69.0 ms | 141.9 s | 350.9 s | 73 |
| 95% | 8 | 16 | 178.0 | 117.4 ms | 239.3 s | 575.6 s | 69 |
| 90% | 8 | 16 | 134.9 | 168.3 ms | 340.7 s | 759.3 s | 59 |
| 85% | 9 | 16 | 105.3 | 225.4 ms | 348.4 s | 972.6 s | 62 |
| 80% | 9 | 16 | 88.4 | 290.1 ms | 446.3 s | 1158.9 s | 52 |
| 75% | 10 | 16 | 74.7 | 362.4 ms | 556.7 s | 1370.9 s | 53 |
| 70% | 11 | 16 | 65.8 | 445.9 ms | 683.3 s | 1557.5 s | 47 |
| 65% | 12 | 16 | 57.9 | 541.9 ms | 565.8 s | 1769.9 s | 49 |
| 60% | 13 | 16 | 52.4 | 651.8 ms | 676.9 s | 1954.4 s | 37 |
| 55% | 14 | 16 | 47.2 | 783.9 ms | 805.8 s | 2169.6 s | 37 |
| 50% | 16 | 16 | 43.5 | 938.1 ms | 953.9 s | 2354.0 s | 37 |
| 45% | 17 | 16 | 39.9 | 1130.1 ms | 1078.2 s | 2568.0 s | 33 |
| 40% | 19 | 16 | 37.2 | 1363.4 ms | 898.4 s | 2750.7 s | 26 |
| 35% | 22 | 16 | 34.5 | 1694.6 ms | 883.5 s | 2966.5 s | 28 |
| 30% | 25 | 16 | 32.5 | 2080.1 ms | 1091.4 s | 3155.6 s | 26 |
| 25% | 32 | 16 | 30.5 | 2681.1 ms | 3.1 s | 3363.9 s | 29 |
| 20% | 39 | 16 | 28.8 | 3511.7 ms | 3.1 s | 3562.5 s | 33 |
| 15% | 51 | 16 | 27.2 | 4908.5 ms | 3.3 s | 3766.0 s | 46 |
| 10% | 73 | 16 | 25.9 | 7029.0 ms | 3.2 s | 3960.3 s | 26 |

DRAM/SSD capacity is infinite in this model. Implied peak occupancy at S=1024 (not enforced): **30% → 31.3 GiB DRAM + 12.5 GiB SSD**; **10% → 117.3 GiB DRAM + 47.1 GiB SSD**. Full table in [`kv_working_set_test_report.md`](kv_working_set_test_report.md) §8. Σ fetch at 10% is 3872 s vs 3960 s makespan.

Regenerate figures:

```bash
python3.11 kv_working_set_test_report/gpu_frac_sweep/run_sweep.py
python3.11 kv_working_set_test_report/gpu_frac_sweep/plot_gpu_frac.py
```
