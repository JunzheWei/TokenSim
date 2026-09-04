# KV Working-Set Offload Test Report / KV 工作集分层测试报告

Date / 日期: 2026-09-04  
Simulator / 模拟器: TokenSim (roofline backend, Python 3.11)  
Primary arrival / 主测试到达: `--distribution burst`

This report documents the 70B / H200 comparison of **all-GPU decode** vs **hierarchical KV working-set fetch** (GPU 30% / DRAM 50% / SSD 20%). Burst is the official case so `--qps` does not set inter-arrival time. Fetch is **page-fault**: the first decode faults the cold tail once; later steps only pay for tokens that newly slide off GPU.

本报告记录 LLaMa2-70B + 1×H200 上 **全 GPU decode** 与 **分层 KV 工作集读取**（GPU 30% / DRAM 50% / SSD 20%）的对照。正式用例使用 **burst**。Fetch 为 **缺页**：第一次 decode 把冷尾读入一次，之后只为新滑出 GPU 的 token 付 I/O。

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

Working-set media (`data/kv_working_set/hier_30_50_20.json`):

| Tier | Fraction | Fixed latency | Bandwidth | Used in v1 latency? |
| --- | --- | --- | --- | --- |
| GPU / HBM | 0.3 (newest tokens) | 0 µs (metadata) | 2000 GB/s | No — still roofline |
| DRAM | 0.5 | **2 µs** (PCIe DMA) | **50 GB/s** | Yes |
| SSD | 0.2 (oldest) | **100 µs** (NVMe) | **7 GB/s** | Yes |

v1 decode step: `T_step = T_roofline + T(dram_miss) + T(ssd_miss)` (page-fault, not a full cold-set read every token). Prefill / recompute do **not** add fetch. GPU occupancy **is** reduced by `gpu_frac`: only `floor(S * gpu_frac)` tokens charge HBM blocks (see §8).

v1 decode：缺页读取，不是每步把冷 KV 全量再读。Prefill / 重算不加 fetch。`gpu_frac` **会减少** GPU KV 占用：HBM 只按 `floor(S * gpu_frac)` 计块（见 §8）。

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

**Practical minimum for this test (one in-flight 70B request ~1k context):** about **123 GiB** HBM. **Minimum for the process to start:** just over **120.76 GiB**. H200 141 GiB is enough for weights + a short KV window, **not** for 100 concurrent full contexts — both arms preempted ~73 times.

对本测试「同时保住一条 ~1k 上下文的 70B 请求」，大约需要 **123 GiB**。进程能启动的下限是刚超过 **120.76 GiB**。H200 141 GiB 装得下权重和一小段 KV，**装不下** 100 条并发满上下文，所以两臂都会抢占（约 73 次）。

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
  --results_path /tmp/tokensim_kv_ws_cmp/all_gpu_burst
```

### Case 2 (official) — burst, hierarchical 30/50/20 / 正式用例：burst 分层

Same arrival and cluster. Page-fault DRAM/SSD reads on decode (cold tail once, then window slide). **Current code also caps GPU blocks at `gpu_frac=0.3`.** The JSON in §5 was captured **before** occupancy capping (73 preemptions, 287 tok/s). Re-run numbers match §8 `gpu_frac=30%` (46 preemptions, 707 tok/s).

到达与集群相同。Decode 缺页读 DRAM/SSD。**当前代码还会按 `gpu_frac=0.3` 限制 GPU 块。** §5 的 JSON 是占用限制之前的快照（73 次抢占，287 tok/s）。重跑应与 §8 的 30% 行一致（46 次抢占，707 tok/s）。

```bash
python3.11 ./benchmark.py --batching paged-attn --qps 10 \
  --distribution burst \
  --cluster ./data/clusters/1_h200/h1.json \
  --model ./data/psla/llama-70b.json \
  --kv_working_set_config ./data/kv_working_set/hier_30_50_20.json \
  --verbose none \
  --results_path /tmp/tokensim_kv_ws_cmp/hier_burst
```

### Case 3 (optional) — uniform QPS 10 / 可选：均匀到达 QPS=10

Same model/hardware, `--distribution uniform` (CLI default). Inter-arrival = `1/10 = 0.1 s`. Offered 10 r/s is far above achieved ~0.28 / ~0.025 r/s, so the system is still saturated. Numbers are almost the same as burst; kept to show that **`--qps 10` does not mean 10 completed requests per second**.

同一模型与硬件。到达间隔 0.1 s。注入 10 r/s 远高于完成率，系统仍然饱和。数值与 burst 几乎相同，用来说明 **`--qps 10` 不是完成 10 r/s**。

```bash
python3.11 ./benchmark.py --batching paged-attn --qps 10 \
  --cluster ./data/clusters/1_h200/h1.json \
  --model ./data/psla/llama-70b.json --verbose none

python3.11 ./benchmark.py --batching paged-attn --qps 10 \
  --cluster ./data/clusters/1_h200/h1.json \
  --model ./data/psla/llama-70b.json \
  --kv_working_set_config ./data/kv_working_set/hier_30_50_20.json \
  --verbose none
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

## 5. Official results (burst, page-fault fetch) / 正式结果（burst + 缺页）

| Metric | Case 1 all GPU | Case 2 hierarchical | Ratio (hier / GPU) |
| --- | --- | --- | --- |
| TTFT p50 | 141.88 s | 145.01 s | **1.02×** |
| TTFT p99 | 316.77 s | 323.17 s | **1.02×** |
| TTFT min | **0.871 s** | **0.871 s** | **1.0×** |
| TTFT avg | 146.92 s | 150.03 s | 1.02× |
| TPOT p50 | **69.0 ms** | **70.6 ms** | **1.02×** |
| TPOT p99 | 131.3 ms | 133.5 ms | 1.02× |
| TPOT avg | 82.0 ms | 83.6 ms | 1.02× |
| Per-request 1/TPOT p50 | 14.48 tok/s | 14.16 tok/s | 1/1.02 |
| System token/s | **292.0** | **286.5** | **1/1.02** |
| Stdout prefill token/s | 146.0 | 143.3 | 1/1.02 |
| Achieved r/s | 0.285 | 0.280 | 1/1.02 |
| Simulated duration | 350.91 s | 357.59 s | 1.02× |
| Preemptions / recomputes | 73 / 73 | 73 / 73 | same |
| Σ `kv_ws_fetch_latency` | 0 | **6.68 s** | first-touch + slide |

With page-fault fetch, hierarchical decode is within **~2%** of all-GPU on TTFT, TPOT, and token/s. The remaining gap is the one-time cold-tail fault plus ~1 token/step as the window slides (Σ fetch 6.68 s vs 3718 s when every step re-read the whole cold set).

缺页之后，分层与全 GPU 的 TTFT / TPOT / token/s 相差约 **2%**。剩余差距来自第一次把冷尾读入，以及窗口每次滑出约 1 个 token（Σ fetch 6.68 s；以前每步全量读是 3718 s）。

Previous full-read-every-step (for comparison) / 此前每步全量读（对照）:

| Metric | All GPU | Hierarchical (full read) |
| --- | --- | --- |
| TPOT p50 | 69.0 ms | 814.9 ms (11.8×) |
| System token/s | 292 | 25.2 (1/11.6) |
| Σ fetch | 0 | 3718 s |

### Comparison figures / 数据对比图

PNG + JSON live under `kv_working_set_test_report/` (page-fault burst rerun).

图和 JSON 在 `kv_working_set_test_report/`（缺页后的 burst 重跑）。

**Figure 1 — slowdown on one axis / 同一纵轴上的变慢倍数**

Ratios sit near **1.02×** (was ~11.8× with full-read-every-step).

倍数约 **1.02×**（每步全量读时约 11.8×）。

![Slowdown TTFT p50/p99 and TPOT p50/p99](kv_working_set_test_report/fig_slowdown.png)

**Figure 2 — absolute TTFT and TPOT / 绝对 TTFT 与 TPOT**

Left: TTFT in seconds (queue wait still dominates p50; the two arms nearly overlap). Right: TPOT in milliseconds (~69 vs ~71 ms).

左：TTFT（秒），p50 仍是排队，两臂几乎重合。右：TPOT（毫秒），约 69 vs 71 ms。

![TTFT seconds and TPOT milliseconds grouped bars](kv_working_set_test_report/fig_ttft_tpot.png)

**Figure 3 — token/s / 吞吐**

Same 102,452 tokens; duration 351 s vs 358 s. System token/s **292 vs 286.5**.

两边仍是 102,452 token；时长 351 s vs 358 s。系统 token/s **292 vs 286.5**。

![System, stdout, and per-request token/s](kv_working_set_test_report/fig_tokens.png)

### Optional Case 3 — historical uniform QPS=10 with full-read fetch / 历史：均匀 QPS=10 + 每步全量读

Recorded before page-fault. Not comparable to §5.

缺页改动前的数据，不能与 §5 直接比。

| Metric | All GPU | Hierarchical (full read) |
| --- | --- | --- |
| Offered QPS | 10 | 10 |
| Achieved r/s | 0.285 | 0.025 |
| TTFT p50 | 137.68 s | 1670.89 s |
| TPOT p50 | 69.0 ms | 816.3 ms |
| System token/s | 291.9 | 25.2 |

---

## 6. How to read token/s / 如何读 token/s

Use JSON **`output_token_ps`** for cluster throughput (prefill+decode). The terminal line **omits decode tokens** (~half of this workload).

集群吞吐看 JSON 的 **`output_token_ps`**。终端 `Thoughput ... token/s` **不含 decode token**（本负载大约少一半）。

Per-request generation speed is `1000 / TPOT_ms` (14.48 vs 14.16 tok/s after page-fault). That is **not** 292 tok/s: the worker interleaves other requests’ prefills and recomputes.

单请求生成速度是 `1000 / TPOT_ms`（缺页后 14.48 vs 14.16 tok/s）。这不是 292 tok/s：worker 会穿插别人的 prefill 和重算。

---

## 7. What page-fault changed / 缺页改了什么

Full-read-every-step charged `T(dram)+T(ssd)` for the whole cold set on **every** decode token (~12× slower). Page-fault keeps a per-request watermark `kv_ws_fetched_end`: storage is `[0, gpu_start)`; only `[fetched_end, gpu_start)` is I/O.

每步全量读会对每个 decode token 收取整段冷 KV，大约慢 12 倍。缺页在请求上保留水位 `kv_ws_fetched_end`：存储区是 `[0, gpu_start)`，只对尚未读过的 `[fetched_end, gpu_start)` 计时。

After this change, hierarchical **fetch-only** (occupancy still full GPU) nearly matched all-GPU speed (~2%). Capping GPU blocks with `gpu_frac` is the occupancy sweep in §8: 30/50/20 then reaches **707 tok/s** and 46 preemptions, not the fetch-only 287 tok/s / 73 preemptions in §5.

缺页之后，若占用仍按全 GPU 计，分层与全 GPU **几乎追平**（约 2%）。用 `gpu_frac` 限制 GPU 块之后见 §8：30/50/20 变为 **707 tok/s**、抢占 46，而不是 §5 里只加 fetch 的 287 tok/s / 73 次抢占。

---

## 8. gpu_frac occupancy sweep / 占用扫描

`gpu_frac` now charges only `floor(S * gpu_frac)` tokens of GPU blocks (`gpu_resident_blocks` in `BlockManager`). Remainder of `(1 - gpu_frac)` is split DRAM:SSD = 5:2, same media as `hier_30_50_20.json`. Burst recipe unchanged. DRAM/SSD **capacity** is still infinite.

`gpu_frac` 现在只把 `floor(S * gpu_frac)` 个 token 计入 GPU block。其余按 DRAM:SSD = 5:2 拆分。Burst 与正式用例相同。DRAM/SSD **容量**仍无限。

Peak decode concurrency at S=1024 is leftover blocks after watermark (`513`) / `ceil(floor(1024*gpu_frac)/16)`.

| gpu_frac | Peak B (S=1024) | token/s | vs 100% | TPOT p50 | per-user | TTFT p50 | makespan | preempt | Σ fetch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 292.0 | 1.00× | 69.0 ms | 14.5 | 141.9 s | 350.9 s | 73 | 0 |
| 95% | 8 | 292.6 | 1.00× | 69.8 ms | 14.3 | 142.3 s | 350.1 s | 72 | 0.52 s |
| 90% | 8 | 320.7 | 1.10× | 70.3 ms | 14.2 | 142.0 s | 319.4 s | 64 | 0.99 s |
| 85% | 9 | 321.2 | 1.10× | 71.0 ms | 14.1 | 110.2 s | 318.9 s | 65 | 1.47 s |
| 80% | 9 | 351.5 | 1.20× | 72.0 ms | 13.9 | 111.4 s | 291.5 s | 65 | 1.94 s |
| 75% | 10 | 352.9 | 1.21× | 72.9 ms | 13.7 | 112.4 s | 290.3 s | 66 | 2.41 s |
| 70% | 11 | 390.0 | 1.34× | 74.2 ms | 13.5 | 112.8 s | 262.7 s | 63 | 2.89 s |
| 65% | 12 | 391.0 | 1.34× | 75.3 ms | 13.3 | 79.9 s | 262.0 s | 65 | 3.36 s |
| 60% | 13 | 438.5 | 1.50× | 76.8 ms | 13.0 | 80.7 s | 233.7 s | 64 | 3.84 s |
| 55% | 14 | 441.8 | 1.51× | 78.9 ms | 12.7 | 81.3 s | 231.9 s | 64 | 4.32 s |
| 50% | 16 | 503.7 | 1.73× | 80.8 ms | 12.4 | 82.1 s | 203.4 s | 57 | 4.78 s |
| 45% | 17 | 503.7 | 1.73× | 83.0 ms | 12.0 | 78.3 s | 203.4 s | 63 | 5.26 s |
| 40% | 19 | 589.1 | 2.02× | 86.0 ms | 11.6 | 47.8 s | 173.9 s | 54 | 5.74 s |
| 35% | 22 | 589.9 | 2.02× | 89.7 ms | 11.1 | 48.9 s | 173.7 s | 57 | 6.22 s |
| **30%** | **25** | **706.6** | **2.42×** | 93.4 ms | 10.7 | **2.94 s** | 145.0 s | 46 | 6.68 s |
| 25% | 32 | 712.3 | 2.44× | 99.9 ms | 10.0 | 3.51 s | 143.8 s | 50 | 7.16 s |
| 20% | 39 | 828.1 | 2.84× | 109.9 ms | 9.1 | 4.20 s | 123.7 s | 41 | 7.64 s |
| 15% | 51 | 899.2 | 3.08× | 128.3 ms | 7.8 | 5.77 s | 113.9 s | 49 | 8.12 s |
| 10% | 73 | 1063.5 | 3.64× | 144.1 ms | 6.9 | 5.77 s | 96.3 s | 26 | 8.59 s |

**EN.** System token/s and TTFT improve because more requests share the GPU; TPOT and per-user tok/s worsen because decode batches are larger (attention is summed) and fetch grows. Σ fetch is still small vs makespan. The staircase (95≈100, 90≈85, …) is GPU **block** rounding, not 5% itself.

**中文。** 系统 token/s 和 TTFT 变好是因为并发上去、排队缩短；TPOT 和单流变差是因为 decode batch 变大（attention 逐条相加）以及 fetch 增加。Σ fetch 相对 makespan 仍然很小。台阶来自 **block** 取整，不是 5% 本身有特殊意义。

At `gpu_frac=15%` and `10%`, TTFT p50 = p99 = 5.77 s: all 100 prompts fit in the first packed prefill.

`gpu_frac=15%` 和 `10%` 时 TTFT p50=p99=5.77 s：100 条 prompt 都能进第一波 packed prefill。

Figures: [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) (token/s chart includes peak B).

```bash
python3.11 kv_working_set_test_report/gpu_frac_sweep/run_sweep.py
python3.11 kv_working_set_test_report/gpu_frac_sweep/plot_gpu_frac.py
```

---

## 9. Reproduce / 复现

Python 3.11, repo root. Official two-arm commands in §3. Occupancy sweep in §8.

| Artifact | Path |
| --- | --- |
| Burst all-GPU JSON | [`kv_working_set_test_report/all_gpu_burst.json`](kv_working_set_test_report/all_gpu_burst.json) |
| Burst hierarchical JSON (fetch-only, pre-occupancy) | [`kv_working_set_test_report/hier_burst.json`](kv_working_set_test_report/hier_burst.json) |
| gpu_frac inference-speed MD | [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) |
| Figure 1–3 PNG | [`fig_slowdown.png`](kv_working_set_test_report/fig_slowdown.png), [`fig_ttft_tpot.png`](kv_working_set_test_report/fig_ttft_tpot.png), [`fig_tokens.png`](kv_working_set_test_report/fig_tokens.png) |
| Canvas (fetch-only 30/50/20) | [TTFT / TPOT / token/s](/home/kewei/.cursor/projects/home-kewei-projects-TokenSim/canvases/ttft-tpot-working-set.canvas.tsx) |
| Canvas (occupancy sweep) | [gpu_frac occupancy sweep](/home/kewei/.cursor/projects/home-kewei-projects-TokenSim/canvases/gpu-frac-occupancy-sweep.canvas.tsx) |

设计说明见 [offload_sim.md](offload_sim.md)。
