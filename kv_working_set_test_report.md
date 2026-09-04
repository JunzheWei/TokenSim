# KV Working-Set Offload Test Report / KV 工作集分层测试报告

Date / 日期: 2026-09-04  
Simulator / 模拟器: TokenSim (roofline backend, Python 3.11)  
Primary arrival / 主测试到达: `--distribution burst`

This report documents the 70B / H200 comparison of **all-GPU decode** vs **hierarchical KV working-set fetch** (GPU 30% / DRAM 50% / SSD 20%). Burst is the official case so `--qps` does not set inter-arrival time.

本报告记录 LLaMa2-70B + 1×H200 上 **全 GPU decode** 与 **分层 KV 工作集读取**（GPU 30% / DRAM 50% / SSD 20%）的对照。正式用例使用 **burst**：所有请求在仿真时刻 0 同时到达，`--qps` 不再控制间隔。

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

v1 decode step: `T_step = T_roofline + T(dram) + T(ssd)`. Prefill / recompute do **not** add fetch. GPU occupancy is **not** reduced by `gpu_frac`.

v1 decode：`T_step = T_roofline + T(dram) + T(ssd)`。Prefill / 重算不加 fetch。`gpu_frac` **不减少** GPU KV 占用。

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

Same arrival and cluster. Adds blocking DRAM/SSD fetch on every decode step.

到达与集群相同。每个 decode 步叠加 DRAM/SSD 阻塞读取。

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

## 5. Official results (burst) / 正式结果（burst）

| Metric | Case 1 all GPU | Case 2 hierarchical | Ratio (hier / GPU) |
| --- | --- | --- | --- |
| TTFT p50 | 141.88 s | 1669.15 s | **11.8×** |
| TTFT p99 | 316.77 s | 3733.99 s | **11.8×** |
| TTFT min | **0.871 s** | **0.871 s** | **1.0×** |
| TTFT avg | 146.92 s | 1728.13 s | 11.8× |
| TPOT p50 | **69.0 ms** | **814.9 ms** | **11.8×** |
| TPOT p99 | 131.3 ms | 1565.6 ms | 11.9× |
| TPOT avg | 82.0 ms | 965.4 ms | 11.8× |
| Per-request 1/TPOT p50 | 14.5 tok/s | 1.23 tok/s | 1/11.8 |
| System token/s | **292.0** | **25.2** | **1/11.6** |
| Stdout prefill token/s | 146.0 | 12.6 | 1/11.6 |
| Achieved r/s | 0.285 | 0.025 | 1/11.6 |
| Simulated duration | 350.91 s | 4068.56 s | 11.6× |
| Preemptions / recomputes | 73 / 73 | 73 / 73 | same order |
| Σ `kv_ws_fetch_latency` | 0 | 3717.65 s | decode only |

Min TTFT is identical: the first packed prefill is the same compute. p50/p99 TTFT grow because the single hybrid worker is busy; hierarchical decode steps include `T_fetch`, so the queue drains ~12× slower. **TPOT is the direct working-set cost.** System token/s falls by the duration ratio because token counts match.

TTFT min 相同：第一批 packed prefill 计算一样。p50/p99 变大是因为唯一 hybrid worker 被占住；分层 decode 含 `T_fetch`，队列慢约 12 倍。**TPOT 才是工作集的直接代价。** 两边 token 数相同，系统 token/s 随仿真时长同比下降。

### Comparison figures / 数据对比图

Figures are stored next to this report so they remain in the repo (not only in chat). Data: `kv_working_set_test_report/all_gpu_burst.json`, `hier_burst.json`.

对比图与 JSON 保存在报告同目录，不依赖会话里的临时图。

**Figure 1 — slowdown on one axis / 同一纵轴上的变慢倍数**

TTFT (seconds) and TPOT (milliseconds) cannot share a raw y-axis. The ratio *hierarchical / all-GPU* puts both on one plot. All four bars sit near **11.8×**.

TTFT（秒）和 TPOT（毫秒）不能画在同一根原始纵轴上。用 *分层 / 全 GPU* 倍数可以把两个指标画在一张图里。四根柱都在 **11.8×** 附近。

![Slowdown TTFT p50/p99 and TPOT p50/p99](kv_working_set_test_report/fig_slowdown.png)

**Figure 2 — absolute TTFT and TPOT / 绝对 TTFT 与 TPOT**

Left: TTFT in seconds (queue wait dominates p50/p99). Right: TPOT in milliseconds (direct fetch addend).

左：TTFT（秒），p50/p99 主要是排队。右：TPOT（毫秒），工作集读取的直接加项。

![TTFT seconds and TPOT milliseconds grouped bars](kv_working_set_test_report/fig_ttft_tpot.png)

**Figure 3 — token/s / 吞吐**

Same 102,452 tokens; duration 351 s vs 4069 s. Use **System (prefill+decode)** as cluster throughput. Stdout omits decode tokens. Per-request `1/TPOT p50` is one-stream generation speed (14.5 vs 1.23 tok/s).

两边都是 102,452 token；时长 351 s vs 4069 s。集群吞吐看 **System**。终端 stdout 不含 decode token。`1/TPOT p50` 是单请求生成速度（14.5 vs 1.23 tok/s）。

![System, stdout, and per-request token/s](kv_working_set_test_report/fig_tokens.png)

### Optional Case 3 (uniform QPS=10) — sanity check / 可选用例核对

| Metric | All GPU | Hierarchical |
| --- | --- | --- |
| Offered QPS | 10 | 10 |
| Achieved r/s | 0.285 | 0.025 |
| TTFT p50 | 137.68 s | 1670.89 s |
| TPOT p50 | 69.0 ms | 816.3 ms |
| System token/s | 291.9 | 25.2 |
| Preemptions | 67 | 69 |

Burst vs uniform QPS=10: TPOT and system token/s differ by <1%. Arrival pattern is not the reason hierarchical is slower. `--qps 10` only packs arrivals into the first ~10 s; service takes hundreds to thousands of seconds.

burst 与 uniform QPS=10：TPOT 与系统 token/s 相差不到 1%。分层变慢不是到达模式造成的。`--qps 10` 只是把到达挤在前约 10 秒；服务要几百到几千秒。

---

## 6. How to read token/s / 如何读 token/s

Use JSON **`output_token_ps`** for cluster throughput (prefill+decode). The terminal line **omits decode tokens** (~half of this workload).

集群吞吐看 JSON 的 **`output_token_ps`**。终端 `Thoughput ... token/s` **不含 decode token**（本负载大约少一半）。

Per-request generation speed is `1000 / TPOT_ms` (14.5 vs 1.23 tok/s). That is **not** 292 tok/s: the worker interleaves other requests’ prefills and recomputes.

单请求生成速度是 `1000 / TPOT_ms`（14.5 vs 1.23 tok/s）。这不是 292 tok/s：worker 会穿插别人的 prefill 和重算。

---

## 7. Why hierarchical does not beat all-GPU / 为何分层没有快过全 GPU

v1 **adds I/O without shrinking GPU KV**. Both arms hit the same ~20 GiB KV ceiling and preempt. Hierarchical extra cost is re-reading all DRAM+SSD tokens **every decode step**:

v1 **只加 I/O、不减 GPU KV**。两臂都撞上约 20 GiB KV 上限并抢占。分层额外代价是 **每个 decode 步把 DRAM+SSD 上的 token 整段再读一遍**：

```text
T_fetch = lat_dram + bytes_dram / 50 GB/s + lat_ssd + bytes_ssd / 7 GB/s
bytes_*  = floor(S × frac) × 2.50 MiB
```

A benefit vs all-GPU needs a workload all-GPU **cannot** hold without thrashing, and a model that actually keeps only `gpu_frac` of KV on HBM (not implemented in v1). Burst is the right stress arrival; it does not by itself create an offload win.

要体现 offload 好处，需要全 GPU 会因显存不够而重算的负载，并且模拟器真正只在 HBM 上保留 `gpu_frac` 的 KV（v1 未做）。Burst 适合压测到达，本身不会让 offload 赢。

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
