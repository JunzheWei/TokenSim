# KV Working-Set Offload Test Report / KV 工作集分层测试报告

Date / 日期: 2026-09-04  
Simulator / 模拟器: TokenSim (roofline backend, Python 3.11)  
Primary arrival / 主测试到达: `--distribution burst`

This report documents the 70B / H200 comparison of **all-GPU decode** vs **hierarchical KV working-set fetch** (GPU 30% / DRAM 50% / SSD 20%, N3X-SLC 4K / `qd_cap=32`). Burst is the official case so `--qps` does not set inter-arrival time. Fetch is **page-fault**: the first decode faults the cold tail once; later steps only pay for tokens that newly slide off GPU. `gpu_frac` also caps GPU KV occupancy.

本报告记录 LLaMa2-70B + 1×H200 上 **全 GPU decode** 与 **分层 KV 工作集读取**（GPU 30% / DRAM 50% / SSD 20%，N3X-SLC 4K / `qd_cap=32`）的对照。正式用例使用 **burst**。Fetch 为 **缺页**：第一次 decode 把冷尾读入一次，之后只为新滑出 GPU 的 token 付 I/O。`gpu_frac` 同时限制 GPU KV 占用。

---

## 1. Hardware and model / 硬件与模型

| Item / 项 | Value / 值 |
| --- | --- |
| Cluster | `data/clusters/1_h200/h1.json` — 1 hybrid worker, **H200** |
| HBM | **141 GiB**, 1 card (`MM_Card_Num=1`, `Capacity=141`) |
| Compute / BW | 1979 TFLOPS, 4.8 TB/s HBM |
| Parallelism | TP=PP=DP=1 |
| Model JSON | `data/psla/llama-70b.json` → roofline name **`LLaMa2-70B`** |
| Architecture | 80 layers, `Dmodel=8192`, 64 heads (MHA, not GQA), `Max_Token=4096` |
| Workload | 100 synthetic requests, prefill **512±32**, decode **512±32**, `block_size=16` |
| Batching | `paged-attn` |

Working-set media (`data/kv_working_set/hier_30_50_20.json` = `hier_n3x_slc.json`):

| Tier | Fraction | Fixed latency | Bandwidth | Queue | Used in v1 latency? |
| --- | ---: | ---: | ---: | --- | --- |
| GPU / HBM | 0.3 (newest) | 0 µs (metadata) | 2000 GB/s | — | No — still roofline |
| DRAM | 0.5 | **2 µs** (PCIe DMA) | **50 GB/s** | coalesced (`io_size=0`) | Yes |
| SSD | 0.2 (oldest) | **13 µs** 4K@QD1 (N3X-SLC) | **14 GB/s** | 4K, `qd_cap=32` | Yes |

Other shipped SSDs (same 30/50/20, DRAM unchanged): `hier_n3x.json` MLC **18 µs**, `hier_n3.json` N3 **50 µs**. All three use 14 GB/s. See §9.

v1 decode step: `T_step = T_roofline + T(dram_miss) + T(ssd_miss)` (page-fault). Prefill / recompute do **not** add fetch. GPU occupancy **is** reduced by `gpu_frac`: only `floor(S * gpu_frac)` tokens charge HBM blocks (see §8). SSD with `io_size_bytes=4096` uses `T = max(n_ios × L(QD) / QD, bytes / BW)`.

v1 decode：缺页读取。Prefill / 重算不加 fetch。`gpu_frac` **会减少** GPU KV 占用。SSD 按 4K 命令排队。

---

## 2. Minimum GPU memory for 70B / 70B 最小显存

TokenSim `CacheConfig` (TP=1, FP16-style `×2` on the param estimate):

```text
W = (12 × Nlayer × Dmodel² + 50000 × Dmodel) × 2
  = (12 × 80 × 8192² + 50000 × 8192) × 2
  = 129,668,218,880 B  =  120.763 GiB

size_per_token = n_kv_heads × head_dim × 2(K/V) × 2(bytes) × Nlayer
               = 64 × 128 × 2 × 2 × 80
               = 2,621,440 B  =  2.50 MiB / token
```

GPU bytes = `MM_Card_Num × Capacity × 2^30`. If `gpu_bytes <= W`, the engine raises `ConfigurationError` and **never starts**.

GPU 字节 = `MM_Card_Num × Capacity × 2^30`。若 `gpu_bytes <= W`，直接 `ConfigurationError`，**仿真起不来**。

| Meaning / 含义 | Memory / 显存 | Note / 说明 |
| --- | --- | --- |
| **Minimum to load weights / 仅能装权重** | **> 120.763 GiB** | Strictly greater than `W`. A100-40G / A100-80G **cannot** run this 70B config at TP=1 |
| + 1 token KV | 120.765 GiB | Smallest runnable KV |
| + one request S=512 (prefill done) | 122.01 GiB | Typical prompt resident |
| + one request S=1024 (prefill+decode ~512+512) | **123.26 GiB** | One full request in this test |
| + one request S=4096 (`Max_Token`) | 130.76 GiB | Still fits 141 GiB |
| **This H200 leftover for KV** | **141 − 120.763 = 20.237 GiB** | ≈ **8288 tokens** ≈ **518** blocks of 16 |
| Concurrent S=1024 requests in leftover (`gpu_frac=1`) | **8** | 100 burst requests ⇒ GPU cache overflow ⇒ preempt/recompute |

**Practical minimum for this test (one in-flight 70B request ~1k context):** about **123 GiB** HBM. **Minimum for the process to start:** just over **120.76 GiB**. H200 141 GiB is enough for weights and a short KV window, **not** for 100 concurrent full contexts. All-GPU therefore preempts 73 times; hierarchical `gpu_frac=0.3` preempts 46 times (§5).

对本测试「同时保住一条 ~1k 上下文的 70B 请求」，大约需要 **123 GiB**。进程能启动的下限是刚超过 **120.76 GiB**。H200 141 GiB 装得下权重和一小段 KV，**装不下** 100 条并发满上下文。全 GPU 抢占 73 次；分层 `gpu_frac=0.3` 抢占 46 次（§5）。

TP>1 would split `W` across ranks; these numbers are **TP=1**.

---

## 3. Test cases / 测试用例

`--qps` is required by `benchmark.py`. Under **burst**, `benchmark.py` sets `args.qps = inf` before `main()`, so arrival is not 10 r/s. Result files are named `result_inf.json`.

`--qps` 仍是 CLI 必填。**burst** 时会改成 `inf`，到达不再是 10 r/s。结果文件为 `result_inf.json`。

### Case 1 (official) — burst, all GPU / 正式用例：burst 全 GPU

All 100 requests arrive at simulated t=0. No working-set config. Decode latency is roofline only (KV treated as on HBM). Paged-attn still uses the 20 GiB KV budget.

100 个请求在 t=0 同时到达。不传工作集配置。Decode 只走 roofline（KV 当在 HBM）。Paged-attn 仍受 20 GiB KV 预算限制。

```bash
python3.11 ./benchmark.py --batching paged-attn --qps 10 \
  --distribution burst \
  --cluster ./data/clusters/1_h200/h1.json \
  --model ./data/psla/llama-70b.json \
  --verbose none \
  --results_path kv_working_set_test_report/all_gpu_burst
```

### Case 2 (official) — burst, hierarchical 30/50/20 / 正式用例：burst 分层

Same arrival and cluster. Page-fault DRAM/SSD reads on decode (cold tail once, then window slide). **GPU blocks are capped at `gpu_frac=0.3`.** SSD is N3X-SLC 4K / `qd_cap=32` / 14 GB/s.

到达与集群相同。Decode 缺页读 DRAM/SSD。**GPU 块按 `gpu_frac=0.3` 限制。** SSD 为 N3X-SLC 4K。

```bash
python3.11 ./benchmark.py --batching paged-attn --qps 10 \
  --distribution burst \
  --cluster ./data/clusters/1_h200/h1.json \
  --model ./data/psla/llama-70b.json \
  --kv_working_set_config ./data/kv_working_set/hier_30_50_20.json \
  --verbose none \
  --results_path kv_working_set_test_report/hier_burst
```

---

## 4. Metric definitions / 指标定义

| Report name | TokenSim field | Meaning |
| --- | --- | --- |
| **TTFT** | `prefill_time` / `g_time.prefill_time` | Simulated time from last event (arrival for the first step) to first token. Includes **queue wait**. |
| **TPOT** | `decode_time` / `g_time.decode_time` | Mean inter-token interval of decode steps. |
| **System token/s** | JSON `output_token_ps` | `(Σ prefill_len + Σ decode_len) / duration` |
| **Stdout token/s** | printed `Thoughput ... token/s` | `Σ prefill_len / duration` only — **decode tokens omitted** |
| **Per-request decode tok/s** | `1 / TPOT` | One request’s generation rate; not cluster throughput |
| **Offered QPS** | CLI `--qps` (burst → `inf`) | Arrival rate |
| **Achieved r/s** | JSON `output_qps` | `request_count / duration` |

Both official arms processed **102,452** tokens (51,226 prefill + 51,226 decode).

两臂处理的 token 数相同，均为 **102,452**。

---

## 5. Official results (burst, occupancy + SLC 4K) / 正式结果

Re-run 2026-09-04 with current code: `gpu_frac=0.3` occupancy **and** N3X-SLC 4K / `qd_cap=32` / 14 GB/s. All-GPU is unchanged.

本次重跑：占用限制 + SLC 4K。全 GPU 臂与此前一致。

| Metric | Case 1 all GPU | Case 2 hierarchical (SLC) | Ratio (hier / GPU) |
| --- | --- | --- | --- |
| TTFT p50 | 141.88 s | **2.94 s** | **0.021×** |
| TTFT p99 | 316.77 s | 100.69 s | 0.32× |
| TTFT min | **0.871 s** | 2.94 s | 3.37× |
| TTFT avg | 146.92 s | 40.56 s | 0.28× |
| TPOT p50 | **69.0 ms** | **92.6 ms** | **1.34×** |
| TPOT p99 | 131.3 ms | 182.9 ms | 1.39× |
| TPOT avg | 82.0 ms | 105.3 ms | 1.28× |
| Per-request 1/TPOT p50 | 14.48 tok/s | 10.80 tok/s | 0.75× |
| System token/s | **292.0** | **711.1** | **2.44×** |
| Stdout prefill token/s | 146.0 | 355.6 | 2.44× |
| Achieved r/s | 0.285 | 0.694 | 2.44× |
| Simulated duration | 350.91 s | 144.07 s | 0.41× |
| Preemptions / recomputes | 73 / 73 | 46 / 46 | 0.63× |
| Σ `kv_ws_fetch_latency` | 0 | **5.76 s** | cold tail + slide |
| SSD 4K IOs | 0 | 6,616,320 | 10338 tokens × 640 |

Hierarchical **system** token/s is 2.44× all-GPU because occupancy lets ~25 requests share the leftover 20 GiB instead of ~8 (`gpu_frac=1`). TTFT p50 drops from 142 s to 2.94 s (less queueing). TPOT and per-request tok/s get worse: decode batches are larger, and the first decode still page-faults the SSD tail (that cold step is inside the per-request TPOT average). Σ fetch is 5.76 s vs 144 s makespan.

分层系统 token/s 是全 GPU 的 **2.44×**，原因是占用限制提高了并发，不是 SSD 比 HBM 快。TTFT p50 从 142 s 降到 2.94 s。TPOT / 单流变差是因为 decode batch 变大，以及第一次 decode 仍要缺页读 SSD。Σ fetch 5.76 s，相对 144 s 时长仍然很小。

Implied peak DRAM/SSD at this 30% split (S=1024, Peak B=25, **not enforced**): **31.3 GiB + 12.5 GiB** (see §8).

该 30% 拆分的估算峰值占用（S=1024、Peak B=25，**模拟器不限制**）：**31.3 GiB DRAM + 12.5 GiB SSD**（见 §8）。

### Comparison figures / 数据对比图

PNG + JSON: `kv_working_set_test_report/` (this re-run).

**Figure 1 — hierarchical / all-GPU**

TTFT ratios are **below 1** (occupancy cuts queue wait). TPOT ratios are **~1.34×**.

TTFT 倍数 < 1（排队缩短）。TPOT 约 1.34×。

![Slowdown TTFT p50/p99 and TPOT p50/p99](kv_working_set_test_report/fig_slowdown.png)

**Figure 2 — absolute TTFT and TPOT**

Left: TTFT p50 142 s vs 2.94 s. Right: TPOT p50 69 vs 93 ms.

左：TTFT p50。右：TPOT p50。

![TTFT seconds and TPOT milliseconds grouped bars](kv_working_set_test_report/fig_ttft_tpot.png)

**Figure 3 — token/s**

Same 102,452 tokens; duration 351 s vs 144 s. System token/s **292 vs 711**.

两边仍是 102,452 token；时长 351 s vs 144 s。系统 token/s **292 vs 711**。

![System, stdout, and per-request token/s](kv_working_set_test_report/fig_tokens.png)

---

## 6. How to read token/s / 如何读 token/s

Use JSON **`output_token_ps`** for cluster throughput (prefill+decode). The terminal line **omits decode tokens** (~half of this workload).

集群吞吐看 JSON 的 **`output_token_ps`**。终端 `Thoughput ... token/s` **不含 decode token**（本负载大约少一半）。

Per-request generation speed is `1000 / TPOT_ms` (14.48 vs 10.80 tok/s). That is **not** 292 or 711 tok/s: the worker interleaves other requests’ prefills and recomputes.

单请求生成速度是 `1000 / TPOT_ms`（14.48 vs 10.80 tok/s）。这不是系统 token/s：worker 会穿插别人的 prefill 和重算。

---

## 7. What page-fault changed / 缺页改了什么

Full-read-every-step would charge `T(dram)+T(ssd)` for the whole cold set on **every** decode token. Page-fault keeps a per-request watermark `kv_ws_fetched_end`: storage is `[0, gpu_start)`; only `[fetched_end, gpu_start)` is I/O. Official Case 2 uses this plus `gpu_frac=0.3` occupancy and SLC 4K: **711 tok/s**, 46 preemptions (§5).

每步全量读会对每个 decode token 收取整段冷 KV。缺页在请求上保留水位 `kv_ws_fetched_end`：存储区是 `[0, gpu_start)`，只对尚未读过的 `[fetched_end, gpu_start)` 计时。正式 Case 2 是缺页 + 占用 30% + SLC 4K。

---

## 8. gpu_frac occupancy sweep / 占用扫描

`gpu_frac` now charges only `floor(S * gpu_frac)` tokens of GPU blocks (`gpu_resident_blocks` in `BlockManager`). Remainder of `(1 - gpu_frac)` is split DRAM:SSD = 5:2. Burst recipe unchanged. DRAM/SSD **capacity** is still infinite. Media matches official Case 2: DRAM 2 µs / 50 GB/s coalesced; SSD N3X-SLC 13 µs / 14 GB/s / 4K / `qd_cap=32`.

`gpu_frac` 现在只把 `floor(S * gpu_frac)` 个 token 计入 GPU block。其余按 DRAM:SSD = 5:2 拆分。Burst 与正式用例相同。DRAM/SSD **容量**仍无限。介质与正式 Case 2 相同。

Peak decode concurrency at S=1024 is leftover blocks after watermark (`513`) / `ceil(floor(1024*gpu_frac)/16)`.

| gpu_frac | Peak B (S=1024) | token/s | vs 100% | TPOT p50 | per-user | TTFT p50 | makespan | preempt | Σ fetch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 292.0 | 1.00× | 69.0 ms | 14.5 | 141.9 s | 350.9 s | 73 | 0 |
| 95% | 8 | 292.7 | 1.00× | 69.8 ms | 14.3 | 142.3 s | 350.0 s | 72 | 0.43 s |
| 90% | 8 | 320.9 | 1.10× | 70.2 ms | 14.2 | 141.9 s | 319.3 s | 64 | 0.85 s |
| 85% | 9 | 321.4 | 1.10× | 70.9 ms | 14.1 | 110.1 s | 318.7 s | 65 | 1.25 s |
| 80% | 9 | 351.8 | 1.20× | 71.9 ms | 13.9 | 111.3 s | 291.2 s | 65 | 1.66 s |
| 75% | 10 | 353.4 | 1.21× | 72.8 ms | 13.7 | 112.2 s | 289.9 s | 66 | 2.07 s |
| 70% | 11 | 390.6 | 1.34× | 74.1 ms | 13.5 | 112.6 s | 262.3 s | 63 | 2.48 s |
| 65% | 12 | 391.7 | 1.34× | 75.1 ms | 13.3 | 79.7 s | 261.5 s | 65 | 2.89 s |
| 60% | 13 | 439.5 | 1.51× | 76.6 ms | 13.0 | 80.4 s | 233.1 s | 64 | 3.30 s |
| 55% | 14 | 443.0 | 1.52× | 78.7 ms | 12.7 | 81.0 s | 231.3 s | 64 | 3.71 s |
| 50% | 16 | 505.4 | 1.73× | 80.5 ms | 12.4 | 81.8 s | 202.7 s | 57 | 4.12 s |
| 45% | 17 | 505.5 | 1.73× | 82.7 ms | 12.1 | 77.9 s | 202.7 s | 63 | 4.53 s |
| 40% | 19 | 591.9 | 2.03× | 85.5 ms | 11.7 | 47.4 s | 173.1 s | 54 | 4.94 s |
| 35% | 22 | 592.8 | 2.03× | 89.1 ms | 11.2 | 48.5 s | 172.8 s | 57 | 5.35 s |
| **30%** | **25** | **711.1** | **2.44×** | 92.6 ms | 10.8 | **2.94 s** | 144.1 s | 46 | 5.76 s |
| 25% | 32 | 717.3 | 2.46× | 98.7 ms | 10.1 | 3.51 s | 142.8 s | 50 | 6.17 s |
| 20% | 39 | 835.2 | 2.86× | 108.4 ms | 9.2 | 4.20 s | 122.7 s | 41 | 6.58 s |
| 15% | 51 | 908.2 | 3.11× | 126.1 ms | 7.9 | 5.77 s | 112.8 s | 49 | 6.99 s |
| 10% | 73 | 1076.8 | 3.69× | 141.8 ms | 7.1 | 5.77 s | 95.1 s | 26 | 7.40 s |

**EN.** System token/s and TTFT improve because more requests share the GPU; TPOT and per-user tok/s worsen because decode batches are larger (attention is summed) and fetch grows. Σ fetch is still small vs makespan. The staircase (95≈100, 90≈85, …) is GPU **block** rounding, not 5% itself.

**中文。** 系统 token/s 和 TTFT 变好是因为并发上去、排队缩短；TPOT 和单流变差是因为 decode batch 变大（attention 逐条相加）以及 fetch 增加。Σ fetch 相对 makespan 仍然很小。台阶来自 **block** 取整，不是 5% 本身有特殊意义。

At `gpu_frac=15%` and `10%`, TTFT p50 = p99 = 5.77 s: all 100 prompts fit in the first packed prefill.

`gpu_frac=15%` 和 `10%` 时 TTFT p50=p99=5.77 s：100 条 prompt 都能进第一波 packed prefill。

### Peak DRAM / SSD occupancy (not enforced) / 峰值占用（模拟器不限制）

v1 does **not** cap DRAM or SSD. Bytes below are Peak B × per-request split × **2.50 MiB/token** at S=1024 (same Peak B as the table above). This workload has 100 requests, so concurrency cannot exceed 100; decode Peak B stays ≤ 73.

模拟器 **不检查** DRAM/SSD 是否装得下。下表按 decode 满长 S=1024 估算。正式 Case 2（30%）约 **31 GiB DRAM + 13 GiB SSD**。

```text
S_gpu  = floor(S × gpu_frac)
S_dram = floor(S × dram_frac)     # dram_frac = (1 − gpu_frac) × 5/7
S_ssd  = S − S_gpu − S_dram
peak   = Peak_B × S_tier × 2.50 MiB
```

| gpu_frac | Peak B | S_dram | S_ssd | Peak DRAM | Peak SSD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 0 | 0 | 0 | 0 |
| 95% | 8 | 36 | 16 | 0.70 GiB | 0.31 GiB |
| 90% | 8 | 73 | 30 | 1.43 GiB | 0.59 GiB |
| 85% | 9 | 109 | 45 | 2.40 GiB | 0.99 GiB |
| 80% | 9 | 146 | 59 | 3.21 GiB | 1.30 GiB |
| 75% | 10 | 182 | 74 | 4.44 GiB | 1.81 GiB |
| 70% | 11 | 219 | 89 | 5.88 GiB | 2.39 GiB |
| 65% | 12 | 256 | 103 | 7.50 GiB | 3.02 GiB |
| 60% | 13 | 292 | 118 | 9.27 GiB | 3.75 GiB |
| 55% | 14 | 329 | 132 | 11.3 GiB | 4.51 GiB |
| 50% | 16 | 365 | 147 | 14.3 GiB | 5.74 GiB |
| 45% | 17 | 402 | 162 | 16.7 GiB | 6.72 GiB |
| 40% | 19 | 438 | 177 | 20.3 GiB | 8.21 GiB |
| 35% | 22 | 475 | 191 | 25.5 GiB | 10.3 GiB |
| **30%** | **25** | **512** | **205** | **31.3 GiB** | **12.5 GiB** |
| 25% | 32 | 548 | 220 | 42.8 GiB | 17.2 GiB |
| 20% | 39 | 585 | 235 | 55.7 GiB | 22.4 GiB |
| 15% | 51 | 621 | 250 | 77.3 GiB | 31.1 GiB |
| 10% | 73 | 658 | 264 | 117.3 GiB | 47.1 GiB |

Figures: [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) (token/s chart includes peak B).

```bash
python3.11 kv_working_set_test_report/gpu_frac_sweep/run_sweep.py
python3.11 kv_working_set_test_report/gpu_frac_sweep/plot_gpu_frac.py
```

---

## 9. SSD 4K / QD eval / SSD 排队评估

Same burst 70B / H200 / `gpu_frac=0.3`. DRAM stays coalesced 2 µs / 50 GB/s. SSD `read_bw_gbps=14` for all drives. Script: `kv_working_set_test_report/ssd_qos_eval/run_eval.py`.

同一 burst 配方。DRAM 不排队。三盘带宽都是 14 GB/s。

`L(QD)`: `L = L1` for `QD ≤ 32`; `L = L1 × QD / 32` above the knee. For `n_ios > qd_cap ≥ 32` this cancels:

```text
t_iops = n_ios × L(QD) / qd_cap = n_ios × L1 / 32
```

Raising `qd_cap` above 32 therefore **does not** add IOPS; the knee already sets peak command rate `32 / L1`. `qd_cap=8` does serialize (4× the knee `t_iops`). Knee sequential bandwidth `32 / L1 × 4KiB` is ~10.1 GB/s (SLC) and ~2.6 GB/s (N3), both below 14 GB/s, so these cold reads stay IOPS-bound at the knee. That is why 32 / 128 / 512 tie, and why N3 never meets SLC even at `qd_cap=512`.

`qd_cap` 提到 32 以上不会再加快：膝点之后延迟随 QD 线性涨，IOPS 钉在 `32/L1`。只有 `qd_cap=8` 会更串行。

### 9.1 Queued 4K, `qd_cap=32` — SLC > MLC > N3

Official Case 2 is the SLC row (same as `hier_30_50_20.json`).

| Drive | L1 | token/s | vs SLC | TPOT p50 | TTFT p50 | TTFT p99 | Σ fetch | SSD IOs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLC | 13 µs | **711.1** | 1.00× | 92.6 ms | 2.94 s | 100.7 s | **5.76 s** | 6.62e6 |
| MLC | 18 µs | **706.0** | 0.99× | 93.4 ms | 2.94 s | 101.7 s | 6.79 s | 6.62e6 |
| N3 | 50 µs | **675.2** | 0.95× | 98.9 ms | 2.94 s | 108.1 s | 13.41 s | 6.62e6 |

SLC is fastest, then MLC, then N3. The token/s gap is small because each request page-faults the cold tail **once** (10338 SSD tokens, 640×4K each); later decode is almost all DRAM window-slide. Prefill has no fetch, so TTFT p50 is the same (2.94 s). TTFT p99, duration, and token/s move with SSD L1.

排序是 SLC > MLC > N3。每请求只冷读一次，所以 token/s 差距不大。Prefill 不加 fetch，TTFT p50 相同；p99 / 时长 / token/s 随盘的 L1 变化。

![Drive ranking at qd_cap=32](kv_working_set_test_report/ssd_qos_eval/fig_drive_rank.png)

### 9.2 `qd_cap` sweep

| `qd_cap` | SLC tok/s | SLC fetch | N3 tok/s | N3 fetch |
| ---: | ---: | ---: | ---: | ---: |
| **8** | 673.4 | 13.82 s | 560.6 | 44.42 s |
| **32** | 711.1 | 5.76 s | 675.2 | 13.41 s |
| **128** | 711.1 | 5.76 s | 675.2 | 13.41 s |
| **512** | 711.1 | 5.76 s | 675.2 | 13.41 s |

`qd_cap=8` serializes more, so fetch grows and token/s drops. At 32 and above, IOPS stays at the knee (`32 / L1`); SLC and N3 stay apart because their knees differ (~10 GB/s vs ~2.6 GB/s), not because of the 14 GB/s cap.

`qd_cap=8` 更串行，fetch 变大、token/s 下降。32 以上钉在膝点 IOPS；SLC 与 N3 不会打平，因为膝点带宽不同（约 10 GB/s vs 2.6 GB/s）。

![qd_cap sweep](kv_working_set_test_report/ssd_qos_eval/fig_qd_cap.png)

```bash
python3.11 kv_working_set_test_report/ssd_qos_eval/run_eval.py
```

---

## 10. Reproduce / 复现

Python 3.11, repo root. Official two-arm commands in §3. Occupancy sweep in §8. SSD QoS in §9.

| Artifact | Path |
| --- | --- |
| Burst all-GPU JSON | [`kv_working_set_test_report/all_gpu_burst.json`](kv_working_set_test_report/all_gpu_burst.json) |
| Burst hierarchical JSON (occupancy + SLC 4K) | [`kv_working_set_test_report/hier_burst.json`](kv_working_set_test_report/hier_burst.json) |
| SSD QoS summary | [`kv_working_set_test_report/ssd_qos_eval/summary.json`](kv_working_set_test_report/ssd_qos_eval/summary.json) |
| gpu_frac inference-speed MD | [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) |
| Figure 1–3 PNG | [`fig_slowdown.png`](kv_working_set_test_report/fig_slowdown.png), [`fig_ttft_tpot.png`](kv_working_set_test_report/fig_ttft_tpot.png), [`fig_tokens.png`](kv_working_set_test_report/fig_tokens.png) |
| SSD QoS PNG | [`fig_drive_rank.png`](kv_working_set_test_report/ssd_qos_eval/fig_drive_rank.png), [`fig_qd_cap.png`](kv_working_set_test_report/ssd_qos_eval/fig_qd_cap.png) |
| Canvas (official burst) | [TTFT / TPOT / token/s](/home/kewei/.cursor/projects/home-kewei-projects-TokenSim/canvases/ttft-tpot-working-set.canvas.tsx) |
| Canvas (occupancy sweep) | [gpu_frac occupancy sweep](/home/kewei/.cursor/projects/home-kewei-projects-TokenSim/canvases/gpu-frac-occupancy-sweep.canvas.tsx) |

设计说明见 [offload_sim.md](offload_sim.md)。
