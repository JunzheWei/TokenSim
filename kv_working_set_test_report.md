# KV Working-Set Offload Test Report / KV 工作集分层测试报告

Date / 日期: 2026-09-04  
Simulator / 模拟器: TokenSim (roofline backend, Python 3.11)  
Primary arrival / 主测试到达: `--distribution poisson`, per-config **QPS knee**

This report documents the 70B / H200 comparison of **all-GPU decode** vs **hierarchical KV working-set fetch** (GPU 30% / DRAM 50% / SSD 20%, N3X-SLC **128KiB** DMA, `qd_cap=32`, **layer-prefetch** overlap). Each configuration is swept with open-loop Poisson arrivals until the largest stable offered QPS `λ*` (goodput ≥ 0.90 and TTFT p99 ≤ 3× light-load). Fetch is **per-step cold-set read**. Prefill / recompute stay on GPU; after those steps GPU blocks trim to `gpu_frac` and spill writes plus PCIe contention are charged on that step.

本报告记录 LLaMa2-70B + 1×H200 上 **全 GPU decode** 与 **分层 KV 工作集读取**（GPU 30% / DRAM 50% / SSD 20%，N3X-SLC **128KiB** DMA，层间预取）的对照。每种配置用泊松到达扫到稳定膝点 `λ*`。Fetch 为每步读冷 KV。Prefill / 重算在 GPU 上完成；结束后裁到 `gpu_frac` 并计写回与 PCIe 争用。

Burst (100 requests at t=0) previously produced a TTFT p50 cliff from hybrid + prefill-priority scheduling. That is a scheduling artifact, not the official operating point.

Burst（t=0 同时 100 条）曾出现 TTFT p50 悬崖，那是 hybrid + prefill 优先的调度假象，正式结论以膝点为准。

---

## 1. Hardware and model / 硬件与模型

This run uses **MHA-64** (~**2.50 MiB/token**). Production Llama-2-70B is typically **GQA-8** (~**0.31 MiB/token**, **1/8** the KV). I/O volume and throughput drop here are an MHA heavy-load bound. The GQA control with the same recipe is [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md).

本期是 **MHA-64**（约 **2.50 MiB/token**）。真实 LLaMA-2-70B 多为 **GQA-8**。本报告的 I/O 瓶颈与吞吐下降代表 MHA 极端重载；GQA 对照见 [gqa_working_set_test_report.md](gqa_working_set_test_report.md)。

| Item / 项 | Value / 值 |
| --- | ---: |
| Cluster | `data/clusters/1_h200/h1.json` — 1 hybrid worker, **H200** |
| HBM | **141 GiB**, 1 card |
| Compute / BW | 1979 TFLOPS, 4.8 TB/s HBM |
| Parallelism | TP=PP=DP=1 |
| Model JSON | `data/psla/llama-70b.json` → **`LLaMa2-70B`** (MHA-64, 80 layers) |
| Workload | 100 synthetic requests, prefill **512±32**, decode length **= prefill** (not a second draw), `block_size=16` |
| Batching | `paged-attn` |
| Arrival | Poisson; `λ*` search from 0.02 r/s, double then binary, resolution 0.02 |

Working-set media (`data/kv_working_set/hier_30_50_20.json` = `hier_n3x_slc.json`):

| Tier | Fraction | Latency | Bandwidth | Queue | Used? |
| --- | ---: | ---: | ---: | --- | --- |
| GPU / HBM | 0.3 (newest) | — | **4.8 TB/s** (Roofline internal) | — | TransformerRoofline H200 `BW_TBs`; not working-set I/O |
| DRAM | 0.5 | 2 µs | 50 GB/s | coalesced | Yes |
| SSD | 0.2 (oldest) | 13 µs | **14 GB/s** | **128KiB**, `qd_cap=32` | Yes |
| Host PCIe | — | — | 50 GB/s | spill contention | Trim writes |

`hier_30_50_20.json` still carries `"hbm": {"read_bw_gbps": 2000.0}`. That field is a **placeholder**: `fetch_cost` / `spill_cost` only charge DRAM and SSD. GPU-resident KV and matmul use TransformerRoofline H200 (`TFLOPS=1979`, `BW_TBs=4.8`). The JSON entry does **not** throttle HBM to 2000 GB/s.

出厂 JSON 里的 `hbm.read_bw_gbps=2000` 不参与计时。GPU 侧带宽以 Roofline 的 **4.8 TB/s** 为准。

Decode step:

```text
T_fetch = T(dram) + T(ssd)          # full cold set [0, gpu_start)
T_step  = T_fetch/N + max(T_roofline, T_fetch*(N-1)/N)   # N=80
```

After prefill/recompute:

```text
T_spill = max(T_dram_wr, T_ssd_wr, (dram_bytes+ssd_bytes)/pcie_bw)
```

`T_spill` is added to that step so TTFT includes writeback. Official Case 2 Σ spill ≈ **1.80 s** across 100 prompts (~18 ms each).

---

## 2. Minimum GPU memory for 70B / 70B 最小显存

TokenSim `CacheConfig` (TP=1, FP16-style `×2` on the param estimate):

```text
W = (12 × Nlayer × Dmodel² + 50000 × Dmodel) × 2
  = 120.763 GiB

size_per_token = 64 × 128 × 2 × 2 × 80 = 2.50 MiB / token
```

| Meaning / 含义 | Memory / 显存 | Note / 说明 |
| --- | --- | --- |
| **Minimum to load weights** | **> 120.763 GiB** | A100-40G / A100-80G cannot run this 70B config at TP=1 |
| + one request S=1024 | **123.26 GiB** | One full request in this test |
| **H200 leftover for KV** | **20.237 GiB** | ≈ **518** blocks of 16; watermark 1% → 5; avail **513** |
| Concurrent S=1024 at `gpu_frac=1` | **8** (Peak B) | `513 / 64`; Prefill Peak B at S=512 is **16** |
| **Host DRAM / SSD** | derived, **not capped** | Official 30/50/20 Peak B: **31.25 GiB + 12.51 GiB** (§8) |

H200 141 GiB fits weights plus a short KV window, not 100 full contexts. Under Poisson at the all-GPU knee (`λ*=0.12`) Little `N*` is **3.89** and preemptions are **2**, not the burst-100 figure of 73. DRAM/SSD GiB are occupancy after decode trim, not a simulated hard limit.

---

## 3. Test cases / 测试用例

`λ*` is **not a target QPS**. It is searched: Poisson arrival starts at 0.02 r/s, doubles, then binary-searches the **largest stable offered rate** for that config (`output_qps / λ ≥ 0.90` and `ttft_p99 ≤ 3 × ttft_p99(λ=0.02)`). `N*` is then Little's law at that knee using **offered** `λ*` (not `output_qps`): `N* = λ* × request_time.p50`. It is an **estimate** of average in-flight requests (queue + prefill + decode), not a scheduler count and not Peak B. Scripts share `TokenSim/kv_working_set/knee.py`.

`λ*` **不是预期到达率**，是扫出来的：该配置还能稳住的最大到达率。`N*` 是 Little 估计（到达率 × 中位停留时间），不是调度器数出来的并发，也不是 HBM 上限 Peak B。

`kv_working_set_test_report/ssd_qos_eval/run_eval.py` is the **pipeline**, not an SSD-only job. It runs **Case 1**, **Case 2**, the QoS arms in §9, and writes `fig_qps_knee.png` / `fig_ttft_tpot.png`. The file lives under `ssd_qos_eval/` because that directory owns the shared knee harness. Occupancy (§8) is a separate script.

该脚本是 Case 1、Case 2 与 §9 QoS 的总控，并出官方对比图。路径在 `ssd_qos_eval/` 只是评测目录归属，不是“只跑 SSD”。

```bash
python3.11 kv_working_set_test_report/ssd_qos_eval/run_eval.py
```

To replay one reported knee without the search / 只复现已报膝点、不重新搜 `λ*`：

### Case 1 (official) — Poisson knee, all GPU

No working-set config. Result dir `kv_working_set_test_report/all_gpu_poisson`. Reported `λ* = 0.12`.

```bash
python3.11 benchmark.py --batching paged-attn --qps 0.12 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b.json \
  --verbose none --results_path kv_working_set_test_report/all_gpu_poisson
```

### Case 2 (official) — Poisson knee, hierarchical 30/50/20

`--kv_working_set_config data/kv_working_set/hier_30_50_20.json` (128KiB + `layer_prefetch`). Result dir `kv_working_set_test_report/hier_poisson`. Reported `λ* = 0.04`.

```bash
python3.11 benchmark.py --batching paged-attn --qps 0.04 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b.json \
  --verbose none --results_path kv_working_set_test_report/hier_poisson \
  --kv_working_set_config data/kv_working_set/hier_30_50_20.json
```

---

## 4. Metric definitions / 指标定义

| Report name | Meaning |
| --- | --- |
| **λ\*** | Swept, not a target: largest **stable** offered Poisson QPS for this config. / 扫出来的膝点，不是预期 QPS：该配置还能稳住的最大到达率 |
| **N\*** | Little **estimate** at λ\*: `offered_qps × request_time.p50` (queue + prefill + decode). Not a scheduler count; uses offered rate even when goodput is below 1. A float (e.g. 6.37) is an average. / Little 估计，不是数出来的并发 |
| **Peak B** | HBM occupancy **ceiling** at S=1024 after watermark (`513 / ceil(floor(1024×gpu_frac)/16)`). How many decode windows fit, not how many are in flight. / 显存装得下多少条，不是 N* |
| **Peak DRAM / SSD** | Derived host occupancy at Peak B, S=1024. Simulator does **not** cap DRAM or SSD. |
| **TTFT** | `prefill_time` (queue wait + prefill + trim spill) |
| **TPOT** | `decode_time` **p50**: median across requests of each request's **mean** inter-token interval. Tables labeled TPOT p50. / 每条请求先对 decode 步取均值，再对 100 条取中位数 |
| **System token/s** | JSON `output_token_ps` at `λ*` |
| **Σ fetch** | Unoverlapped DRAM+SSD read time (stats); wall clock uses layer-prefetch |

Both arms process **102,452** tokens (51,226 prefill + 51,226 decode). Decode length equals prefill length on every request, so the two sums match.

两臂都是 102,452 token；每条请求的 decode 长度等于 prefill，所以两边和相同。

---

## 5. Official results (Poisson knee, 128KiB + layer prefetch) / 正式结果

| Metric | Case 1 all GPU | Case 2 hierarchical SLC | Ratio (hier / GPU) |
| --- | ---: | ---: | ---: |
| **λ\*** | **0.12** r/s | **0.04** r/s | **0.33×** |
| **N\*** (Little) | **3.89** | **6.37** | 1.64× |
| Peak B (S=1024) | 8 | 25 | 3.13× |
| TTFT p50 | **0.104 s** | **0.185 s** | 1.78× |
| TTFT p99 | **0.159 s** | **0.651 s** | 4.10× |
| TPOT p50 | **63.2 ms** | **311.2 ms** | **4.92×** |
| TPOT p99 | 66.9 ms | 776.7 ms | 11.6× |
| System token/s | **128.4** | **39.9** | **0.31×** |
| Achieved r/s | 0.125 | 0.039 | 0.31× |
| Preemptions | 2 | **0** | — |
| Σ fetch (unoverlapped) | 0 | 2336 s | stats, not wall |
| Σ spill | 0 | **1.80 s** | ~18 ms / prompt |
| SSD IOs (128KiB) | 0 | **1.579e8** | vs 5.053e9 at 4K; count in §7 |

All-GPU fails at 0.14 r/s: output still tracks offered but TTFT p99 jumps to 4.5 s. Hierarchical fails at 0.06 r/s: output QPS plateaus at ~0.043 and TTFT p99 goes to hundreds of seconds. `N*` (6.4) is far below Peak B (25): the SSD/DRAM path saturates before HBM fills.

全 GPU 在 0.14 r/s 因 TTFT p99 超过轻载 3 倍而不稳。分层在 0.06 r/s 吞吐跟不上且排队崩掉。`N*` 小于 Peak B，说明 I/O 先饱和。

Case 2 host occupancy is derived, not capped: Peak B **31.25 GiB DRAM + 12.51 GiB SSD**. The N* row in §8 (**7.96 + 3.19 GiB**) is an **upper bound** (Little N* × full S=1024 decode window), not a measured watermark.

Case 2 host 占用未做硬上限：Peak B **31.25 GiB DRAM + 12.51 GiB SSD**。§8 的 N* 行是上界（假定在飞请求都在 decode 满窗），不是测得的水位。

4K command-amplification control (same 30/50/20, `io_size=4096`): **λ\* = 0.02**, 22.1 tok/s, TPOT p50 **131.9 ms**. 128KiB restores the 14 GB/s roof and doubles stable QPS versus 4K.

![Poisson QPS knee curves](kv_working_set_test_report/fig_qps_knee.png)

![Knee λ*, N*, token/s, TPOT](kv_working_set_test_report/fig_ttft_tpot.png)

---

## 6. How to read token/s / 如何读 token/s

Use JSON **`output_token_ps` at λ\*** for cluster throughput. Light-load 0.02 r/s is ~22 tok/s on both arms (mostly arrival-limited). The knee is where the server stops keeping up.

集群吞吐看膝点上的 **`output_token_ps`**。轻载 0.02 r/s 两臂都约 22 tok/s（到达限制）。膝点才是服务器跟不上的地方。

Per-request generation speed is `1000 / TPOT_ms` (15.8 vs 3.2 tok/s at the knees).

---

## 7. Per-step cold-set fetch and overlap / 每步读冷 KV 与重叠

Full attention needs the whole history each decode token. After trim, GPU keeps `gpu_frac`, so `[0, gpu_start)` is read every step. Layer-prefetch hides fetch under 80-layer compute when I/O is small; when the shared SSD queue fills, `T_step ≈ T_fetch`.

Case 2's **1.579e8** SSD IOs (128KiB) is that full cold-set reread, not a once-per-request page-in. Measured `kv_ws_ssd_read_tokens = 7,895,265` and `2.50 MiB / 128 KiB = 20` IOs/token:

```text
IOs = 100 req × ~512 decode steps × ~154 SSD tokens/step × 20 IOs/token
    ≈ 1.579e8
    = 7,895,265 tokens × 20 IOs/token
```

At 128KiB that is **20.70 TB** (decimal; 18.82 TiB) of SSD read for 100 requests. There is no sparsity or GQA shrinkage in this MHA run. The 4K control is **32×** more commands (**5.053e9** IOs) for the same bytes.

Case 2 的 1.579e8 次 128KiB IO 来自每步全量回读冷 KV，不是每条请求只读一次。无稀疏、无压缩；GQA 会把每 token 字节（因而 IO 次数）缩到 1/8。SSD 读量为 **20.70 TB**（十进制）。

Stats still record unoverlapped fetch (2336 s at the hierarchical knee) so the report can show how much was masked. Wall-clock TPOT is 311 ms, not seconds.

Spill writes after prompt trim are charged (`T_spill ≈ 18 ms` per request). Decode window-slide writes are not.

---

## 8. gpu_frac occupancy sweep / 占用扫描

Same Poisson-knee search at each `gpu_frac` from 100% to 10% (step 5%). Off-GPU remainder DRAM:SSD = 5:2. Media matches Case 2 (128KiB + layer-prefetch). Prefill Peak B at S=512 stays **16**.

Decode 占用随 `gpu_frac` 变；每个点自己扫 `λ*`。Prefill Peak B 仍为 16。

| gpu_frac | Peak B | λ* | N* | token/s | TPOT p50 | TTFT p50 | TTFT p99 | preempt |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | **0.12** | 3.89 | **128.4** | 63.2 ms | 0.10 s | 0.16 s | 2 |
| 95% | 8 | 0.12 | 3.91 | 128.4 | 63.5 ms | 0.11 s | 0.15 s | 1 |
| 90% | 8 | 0.12 | 3.92 | 128.4 | 63.7 ms | 0.11 s | 0.15 s | 0 |
| 85% | 9 | 0.10 | 3.28 | 107.8 | 63.4 ms | 0.10 s | 0.15 s | 0 |
| 80% | 9 | 0.08 | 2.63 | 86.8 | 63.5 ms | 0.10 s | 0.16 s | 0 |
| 75% | 10 | 0.08 | 3.20 | 86.1 | 76.4 ms | 0.12 s | 0.23 s | 0 |
| 70% | 11 | 0.08 | 4.07 | 83.4 | 100.4 ms | 0.11 s | 0.33 s | 0 |
| 65% | 12 | 0.06 | 2.92 | 64.4 | 94.9 ms | 0.12 s | 0.30 s | 0 |
| 60% | 13 | 0.06 | 3.61 | 62.8 | 118.7 ms | 0.12 s | 0.36 s | 0 |
| 55% | 14 | 0.04 | 1.72 | 44.0 | 84.7 ms | 0.12 s | 0.28 s | 0 |
| 50% | 16 | 0.04 | 2.43 | 43.5 | 118.9 ms | 0.13 s | 0.32 s | 0 |
| 45% | 17 | 0.04 | 2.87 | 42.8 | 142.4 ms | 0.13 s | 0.34 s | 0 |
| 40% | 19 | 0.04 | 3.26 | 42.1 | 161.1 ms | 0.13 s | 0.40 s | 0 |
| 35% | 22 | 0.04 | 4.13 | 41.3 | 200.4 ms | 0.14 s | 0.61 s | 0 |
| **30%** | **25** | **0.04** | **6.37** | **39.9** | **311.2 ms** | **0.18 s** | **0.65 s** | 0 |
| 25% | 32 | 0.04 | 9.01 | 38.2 | 430.4 ms | 0.20 s | 0.96 s | 0 |
| 20% | 39 | 0.02 | 1.02 | 22.1 | 100.0 ms | 0.11 s | 0.41 s | 0 |
| 15% | 51 | 0.02 | 1.13 | 22.1 | 110.2 ms | 0.11 s | 0.46 s | 0 |
| 10% | 73 | 0.02 | 1.29 | 22.1 | 129.7 ms | 0.13 s | 0.52 s | 0 |

TPOT p50 is sawtoothed because **each row is taken at that row's own `λ*`**, not at a fixed offered QPS. When `λ*` drops, Little `N*` collapses and SSD queueing disappears, so per-step fetch falls back to light-load. The 25% → 20% step is the clearest: `λ*` 0.04 → 0.02, `N*` **9.01 → 1.02**, TPOT **430.4 → 100.0 ms**. The same effect shows at 70% → 65% (`λ*` 0.08 → 0.06) and 60% → 55% (`λ*` 0.06 → 0.04). Less GPU KV does **not** make a loaded decode step faster; the later rows are simply no longer at the same load.

各行 TPOT 是在各自膝点采集的，不是固定 QPS 对比。`λ*` 下降时并发 `N*` 掉到 ~1，SSD 争用消失，单步 fetch 回到轻载。25%→20% 最明显；不是“留更少显存反而更快”。

Down to ~90% the knee matches all-GPU (`λ*=0.12`, ~128 tok/s): layer-prefetch still hides the small cold set. Below that `λ*` and token/s fall as the cold set grows. `N*` (Little estimate) stays below Peak B (HBM ceiling, not in-flight count). There is **no TTFT p50 cliff** under Poisson.

90% 以上膝点与全 GPU 相同。再往下冷集变大，`λ*` 和 token/s 下降。泊松下没有 burst 那种 TTFT p50 悬崖。

### DRAM / SSD capacity / DRAM 与 SSD 容量

The simulator **does not enforce** DRAM or SSD capacity (only GPU HBM blocks are finite). **极限容量** below is host occupancy **if leftover HBM is packed to Peak B** at decode `S=1024` after trim. That is the size to provision. `λ*` is swept (not a target). `N*` host GiB is **not** the provision number: it is Little N* × a full decode window, an **upper bound**.

模拟器不限制 DRAM/SSD。**极限容量**按 Peak B 配。`N*` 行不是测得的水位，也不是配置目标。

```text
tokens_gpu  = floor(S × gpu_frac)                 # official 30% → 307
tokens_dram = floor(S × dram_frac)                 # official 50% → 512
tokens_ssd  = S − tokens_gpu − tokens_dram         # official 20% → 205
GiB         = PeakB × tokens_tier × size_per_token / 1024   # MiB → GiB; 1 GiB = 1024³
```

Worked example, official 30/50/20:

```text
MHA: size=2.50 MiB, Peak B=25
  DRAM = 25 × 512 × 2.50 / 1024 = 31.25 GiB
  SSD  = 25 × 205 × 2.50 / 1024 = 12.51 GiB

GQA: size=0.3125 MiB, Peak B=205
  DRAM = 205 × 512 × 0.3125 / 1024 = 32.03 GiB
  SSD  = 205 × 205 × 0.3125 / 1024 = 12.83 GiB
```

Prefill keeps the full context on GPU, so these GiB are **decode** only.

#### Peak B 极限容量 (provision this)

| | MHA-64 | GQA-8 | Note / 说明 |
| --- | ---: | ---: | --- |
| `size_per_token` | 2.50 MiB | 0.3125 MiB | GQA = 1/8 |
| HBM leftover | 20.237 GiB | 20.237 GiB | Same weights; GQA only shrinks KV |
| Avail GPU blocks | 513 | 4103 | 1% watermark |
| Official split tokens | 307 / **512** / **205** | same | GPU / DRAM / SSD at S=1024 |
| Peak B (30%) | **25** | **205** | `avail / ceil(307/16)` |
| **DRAM 极限** | **31.25 GiB** | **32.03 GiB** | Peak B × 512 tokens |
| **SSD 极限** | **12.51 GiB** | **12.83 GiB** | Peak B × 205 tokens |
| **建议配置** | **32 GiB DRAM + 13 GiB SSD** | **32 GiB DRAM + 13 GiB SSD** | Same host kit |

GQA Peak B is ~8× (25 → 205) while bytes/token are 1/8, so **host 极限几乎相同**. Do not divide the MHA GiB by 8 to size GQA DRAM/SSD.

GQA 每 token 是 1/8，但 HBM 能装的条数大约也是 8 倍，host **极限对消**。不要把 MHA 的 31+13 GiB 除以 8 给 GQA 配盘。

#### 不是极限：膝点 N* 上界与测试堆积

`N*` is Little's law with **offered** `λ*` (`N* = λ* × request_time.p50`). The GiB row multiplies that N* by a **full** decode window (S=1024, 512/205 tokens). Prefill uses no DRAM/SSD; mid-decode S is shorter. So this is an **upper bound**, not a measured watermark. GQA Case 2 goodput is 0.91, so completed-rate Little would be ~14.6 instead of 15.96. 100 live is a hypothetical pile-up, not Peak B.

`N*` 是 Little 估计（用到达率，不是完成率）。再乘满窗 S=1024 得到 host GiB，是**上界**：prefill 不占 DRAM/SSD，decode 中途更短。配容量用上一张 Peak B 表。

| Basis / 口径 | MHA DRAM | MHA SSD | GQA DRAM | GQA SSD |
| --- | ---: | ---: | ---: | ---: |
| Per request S=1024 | 1.25 GiB | 0.50 GiB | 0.156 GiB | 0.063 GiB |
| **Peak B 极限 (上表)** | **31.25** | **12.51** | **32.03** | **12.83** |
| N* upper bound (S=1024) | 7.96 (N*=6.37) | 3.19 | 2.49 (N*=15.96) | 1.00 |
| 100 live S=1024 | 125.00 | 50.05 | 15.63 | 6.26 |

At this **fixed-concurrency** view (N* or 100 live), GQA host occupancy **is** ~1/8 of MHA. That is not the HBM-packed ceiling.

按固定并发看，GQA 才是 MHA 的约 1/8（并发不再被 Peak B 放大）。

#### gpu_frac 扫描下的极限

Off-GPU remainder DRAM:SSD = 5:2. Pushing `gpu_frac` down raises Peak B, so host 极限跟着涨。Full rows: [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) §7 and [`gqa_gpu_frac_inference_speed.md`](gqa_gpu_frac_inference_speed.md) §7. Companion GQA report: [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md).

| gpu_frac | MHA Peak B | MHA DRAM | MHA SSD | GQA Peak B | GQA DRAM | GQA SSD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 8 | 0 | 0 | 64 | 0 | 0 |
| 50% | 16 | 14.26 GiB | 5.74 GiB | 128 | 14.26 GiB | 5.74 GiB |
| **30%** | **25** | **31.25 GiB** | **12.51 GiB** | **205** | **32.03 GiB** | **12.83 GiB** |
| 10% | 73 | 117.27 GiB | 47.05 GiB | 586 | 117.67 GiB | 47.21 GiB |

At 10% both models need ~**118 GiB DRAM + 47 GiB SSD** if HBM is packed. Official Case 2 stays **32 + 13**.

10% 两边极限都约 118+47 GiB。官方 Case 2 配 **32 + 13** 即可。

Figures: [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md).

```bash
python3.11 kv_working_set_test_report/gpu_frac_sweep/run_sweep.py
python3.11 kv_working_set_test_report/gpu_frac_sweep/plot_gpu_frac.py
```

---

## 9. SSD QoS / SSD 评估

Same Poisson-knee recipe, `gpu_frac=0.3`. DRAM coalesced 2 µs / 50 GB/s. SSD 14 GB/s. Official granularity is **128KiB**.

128KiB sequential DMA sits on the bandwidth roof, so SLC / MLC / N3 and `qd_cap` 8–512 **tie at λ\*=0.04 / 39.9 tok/s** (N3 coalesced light-load TTFT trips the 3× rule at 0.04, so that one arm reports `λ*=0.02`). Drive latency no longer sets the knee.

128KiB 顺序 DMA 打在带宽屋顶上，盘速与 `qd_cap` 不再拉开正式膝点。

4K control (command-amplified): **λ\*=0.02**, 22.1 tok/s — the old IOPS knee.

| Arm | io_size | λ* | token/s | TPOT p50 | TTFT p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| queued SLC 128KiB (Case 2) | 131072 | **0.04** | 39.9 | 311 ms | 0.65 s |
| queued MLC / N3 128KiB | 131072 | 0.04 | 39.9 | 311 ms | 0.65 s |
| coalesced SLC / MLC | 0 | 0.04 | 39.9 | 312 ms | 0.82–0.89 s |
| queued SLC **4K** | 4096 | **0.02** | 22.1 | 131.9 ms | 0.54 s |
| `qd_cap` 8 / 128 / 512 SLC or N3 | 131072 | 0.04 | 39.9 | 311 ms | 0.65 s |

![Drive ranking at the knee](kv_working_set_test_report/ssd_qos_eval/fig_drive_rank.png)

![qd_cap sweep knee QPS](kv_working_set_test_report/ssd_qos_eval/fig_qd_cap.png)

Same pipeline as §3 (Case 1 + Case 2 + these QoS arms):

```bash
python3.11 kv_working_set_test_report/ssd_qos_eval/run_eval.py
```

---

## 10. Next-phase research (not run) / 下阶段（本轮不做）

- **Prefill-Decode 分离:** independent P/D workers plus KV transfer. Expected to remove same-card P/D fighting over HBM; decode nodes still pay per-step cold KV. Existing `data/clusters/8_a100/p2d5.json` and `dispatch_prefill_to_decode` can be reused. Official cluster stays 1×H200 hybrid.
- **GQA 对照 (done):** same recipe on `LLaMa2-70B-GQA` is [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md). Keep `data/psla/llama-70b.json` as MHA-64.

---

## 11. Reproduce / 复现

Python 3.11, repo root.

| Artifact | Path |
| --- | --- |
| All-GPU knee JSON | [`kv_working_set_test_report/all_gpu_poisson.json`](kv_working_set_test_report/all_gpu_poisson.json) |
| Hierarchical knee JSON | [`kv_working_set_test_report/hier_poisson.json`](kv_working_set_test_report/hier_poisson.json) |
| SSD QoS summary | [`kv_working_set_test_report/ssd_qos_eval/summary.json`](kv_working_set_test_report/ssd_qos_eval/summary.json) |
| gpu_frac summary | [`kv_working_set_test_report/gpu_frac_sweep/summary.json`](kv_working_set_test_report/gpu_frac_sweep/summary.json) |
| gpu_frac MD | [`gpu_frac_inference_speed.md`](gpu_frac_inference_speed.md) |
| GQA companion | [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md) |
| Official PNG | [`fig_qps_knee.png`](kv_working_set_test_report/fig_qps_knee.png), [`fig_ttft_tpot.png`](kv_working_set_test_report/fig_ttft_tpot.png) |
| Design | [`offload_sim.md`](offload_sim.md) |

设计说明见 [offload_sim.md](offload_sim.md)。
