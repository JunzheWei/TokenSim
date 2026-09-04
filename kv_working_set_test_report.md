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

v1 decode step: `T_step = T_roofline + T(dram_miss) + T(ssd_miss)` (page-fault, not a full cold-set read every token). Prefill / recompute do **not** add fetch. GPU occupancy is **not** reduced by `gpu_frac`.

v1 decode：缺页读取，不是每步把冷 KV 全量再读。Prefill / 重算不加 fetch。`gpu_frac` **不减少** GPU KV 占用。

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
| Concurrent S=1024 requests in leftover | **8** | 100 burst requests ⇒ GPU cache overflow ⇒ preempt/recompute |

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

Same arrival and cluster. Page-fault DRAM/SSD reads on decode (cold tail once, then window slide).

到达与集群相同。Decode 缺页读 DRAM/SSD（冷尾一次，之后只跟窗口滑动）。

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

After this change, hierarchical **nearly matches** all-GPU speed (~2%). It still does **not** save HBM: both arms preempt 73 times. Offload’s remaining benefit vs all-GPU would be serving a longer context / higher concurrency than 20 GiB KV allows — that needs `gpu_frac` to cap GPU blocks (not in this run).

改完后分层与全 GPU **几乎追平**（约 2%）。显存占用仍未减少，两臂都抢占 73 次。若要在「全 GPU 放不下」时体现 offload 好处，还需要用 `gpu_frac` 限制 GPU 块数（本次未做）。

---

## 8. Reproduce / 复现

Python 3.11, repo root. Commands in §3.

| Artifact | Path |
| --- | --- |
| Burst all-GPU JSON | [`kv_working_set_test_report/all_gpu_burst.json`](kv_working_set_test_report/all_gpu_burst.json) |
| Burst hierarchical JSON | [`kv_working_set_test_report/hier_burst.json`](kv_working_set_test_report/hier_burst.json) |
| Figure 1–3 PNG | [`fig_slowdown.png`](kv_working_set_test_report/fig_slowdown.png), [`fig_ttft_tpot.png`](kv_working_set_test_report/fig_ttft_tpot.png), [`fig_tokens.png`](kv_working_set_test_report/fig_tokens.png) |
| Interactive canvas (same burst numbers) | [TTFT / TPOT / token/s](/home/kewei/.cursor/projects/home-kewei-projects-TokenSim/canvases/ttft-tpot-working-set.canvas.tsx) |

设计说明见 [offload_sim.md](offload_sim.md)。
