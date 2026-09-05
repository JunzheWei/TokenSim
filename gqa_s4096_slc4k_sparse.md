# GQA 末端上下文 4096：稀疏 attention + KV offload（30/0/70）

Date / 日期: 2026-09-05  
Simulator / 模拟器: TokenSim（Roofline 后端，Python 3.11）  
Script / 脚本: [`gqa_working_set_test_report/s4096_sparse/run_sweep.py`](gqa_working_set_test_report/s4096_sparse/run_sweep.py)、[`plot_gpu_frac_sparse.py`](gqa_working_set_test_report/s4096_sparse/plot_gpu_frac_sparse.py)

本文独立，不替换 [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md) / [`offload_sim.md`](offload_sim.md)。

分层改为 **GPU / DRAM / SSD = 30% / 0% / 70%**：冷集全部在 SSD，主机内存不存 KV。用来看 DDR、SSD 的极限容量，以及全量 attention 每步 SSD 回读。

**稀疏注意力可以和 offload 一起用。** `sparse_offload` 把完整 Key/Value 留下（中间不淘汰）；decode **只回读、只计算** sink ∪ window。window=256 已在 GPU 30% 尾部里，所以稀疏 decode **回读仍是 0**。SSD 读取出现在 `hier_full`（每步把 70% 冷集从盘读回）。

`all_gpu_full` 是全量 attention 的**性能**上限，不是质量分数。TokenSim 不生成文本，不算 perplexity / 准确率。

核心问题（性能）：

> 冷集全部放 SSD、不用 DDR 时，稀疏 decode（只 attend 已在 GPU 上的 sink+window）能否避开全量分层的每步 SSD 回读，并提高可稳定的每秒请求数？

四条结论分开写：

1. **all_gpu_full vs hier_full** — 全量 attention 每步从 SSD 回读 70% 冷集的代价  
2. **all_gpu_full vs sparse_offload** — 稀疏 attention + offload 相对全量 GPU 的性能差  
3. **hier_full 的 SSD 读延迟扫描** — 只扫全量分层  
4. **gpu_frac 扫描（dram_frac=0）** — 稀疏 decode 何时打到盘，以及 token/s 怎么变  

---

## 0. 名词与口径

### 0.1 上下文长度 S

**S 是当前请求的上下文长度（token 数）**：`S = prompt 长度 + 已经生成的 decode token 数`。每生成一个 token，S 加 1。

一条请求的时间线：

1. **Prefill** 结束时：S ≈ 2048。  
2. **Decode** 过程中：S 从约 2048 涨到约 4096。  
3. 请求结束时：S ≈ **4096**。实测末端 **4064–4128**（均值 **4096.5**）。

标题里的「4096」是 decode 结束时的最大上下文。

### 0.2 阶段、算法、存储

| 简称 | 全称 | 含义 |
| --- | --- | --- |
| Prefill | 前填 / 首 token 阶段 | 对完整 prompt 做 attention，写出 Key/Value。 |
| Decode | 逐 token 生成阶段 | 每个输出 token 时间只统计这段。 |
| GQA | Grouped Query Attention | 本模型 **GQA-8**。 |
| KV | Key/Value cache | FP16，每 token **0.3125 MiB**。 |
| sink | 注意力汇 | **4** 个，钉在 GPU，参与 decode attention。 |
| window | 滑动窗口 | **256** 个，最新 token。 |
| HBM | GPU 高带宽显存 | 算力与 GPU 驻留 KV。 |
| DRAM / DDR | 主机内存 | 本实验 **0%**：不存 KV。 |
| SSD | 固态盘 | 冷集 **70%**。`hier_full` 每步读；`sparse_offload` 只存、本配方 decode 不读。 |
| SLC | 盘档延迟 | 13 µs、14 GB/s。 |
| 30/0/70 | GPU / DRAM / SSD | 最新 30% 在 GPU，其余 70% 在 SSD。`sparse_offload` 的 GPU 再留 sink。 |
| 冷集 | 不在 GPU 上的 KV | 本实验 = 全部 SSD。 |
| fetch / spill | 回读 / 写出 | decode 读盘；prefill 结束后把洞写到 SSD。 |
| 4K / `qd_cap` | 命令大小 / 队列深度 | 4096 字节，上限 64。 |

### 0.3 并行、占用

| 简称 | 含义 |
| --- | --- |
| TP/PP/DP | 均为 **1**（单卡）。 |
| `paged-attn` | 每块 **16** 个 token。 |
| Peak B | `floor(可用 GPU 块数 / 每请求占用块数)`。 |
| watermark | 4144 块预留 1% → 可用 **4103**。 |

### 0.4 指标

TTFT = 首 token 时间（含 prefill）。TPOT = 每输出 token 时间（不含 prefill）。λ* = 膝点到达率（goodput ≥ 0.90 且 TTFT p99 ≤ 3 × 轻载 TTFT p99）。轻载 λ=0.02。`hier_full` 在 λ=0.02 时 goodput 已低于 0.90，搜索仍回报 0.02 作为下限。

### 0.5 三组对照

| 简称 | 做什么 | 是否评质量 |
| --- | --- | --- |
| all_gpu_full | 全量 KV 在 GPU + 完整上下文 attention | 否 |
| hier_full | 30/0/70 + 完整上下文 attention；70% 每步从 SSD 回读 | 否 |
| sparse_offload | 30/0/70 存完整 KV；decode 只算 sink+window | 否 |

### 0.6 实验配方

| 项 | 值 |
| --- | --- |
| 集群 | 1×H200 |
| 模型 | `LLaMa2-70B-GQA`，FP16 KV **0.3125 MiB/token** |
| 负载 | 100 条，prefill **2048±32**，decode = 该请求 prefill，`random_seed=0` |
| 实测 | 204,826 prefill + 204,826 decode |
| 选中集 | sink **4** + window **256** |
| Offload | **30/0/70** |

```text
全量 attention 峰值并发（S=4096）              = floor(4103 / 256) = 16
hier_full / sparse_offload 峰值并发（30% 尾部）= floor(4103 / 77)  = 53
Prefill 峰值并发（prompt ≈2048）               = floor(4103 / 128) = 32
```

---

## 0.7 DDR / SSD 极限容量

口径：结束时 S=4096，每 token **0.3125 MiB**（整网 80 层）。一条请求完整 KV = **1280 MiB**。  
**极限驻留** = 该层每请求 token × decode 峰值并发 53（GPU 块刚好能塞满 53 条 decode）。这是「要同时撑住 Peak B 条满上下文请求」时，DDR/SSD 至少要备的容量，不是仿真过程中累计读写流量。

| | all_gpu_full | hier_full | sparse_offload |
| --- | ---: | ---: | ---: |
| GPU token / 请求 | 4096 | 1228 | 1232（sink+尾部） |
| DDR token / 请求 | 0 | **0** | **0** |
| SSD token / 请求 | 0 | **2868** | **2864** |
| GPU MiB / 请求 | 1280 | 383.8 | 385.0 |
| **DDR MiB / 请求** | 0 | **0** | **0** |
| **SSD MiB / 请求** | 0 | **896.3** | **895.0** |
| GPU 峰值并发 | 16 | 53 | 53 |
| **DDR 极限** | 0 | **0** | **0** |
| **SSD 极限（×53）** | 0 | **46.4 GiB** | **46.3 GiB** |

对照：若仍用旧的 30/50/20，同样 Peak B=53 时 DDR 极限约 **33.1 GiB**、SSD 约 **13.3 GiB**。30/0/70 把那 33 GiB DDR 全部改到盘上，DDR 极限为 0，SSD 极限约 **46 GiB**。

Prefill 峰值 32 条、每条约 2048 token 全在 GPU：32 × 2048 × 0.3125 = **20.0 GiB**，与可用 4103 块（约 20.0 GiB）对齐。

仿真累计 I/O（100 条请求、不是驻留容量）：

- `hier_full`：SSD **读** 4.404×10⁸ token，回读合计 **9600 s**；写出 47.0 GiB  
- `sparse_offload`：SSD **读 0**；写出 46.9 GiB（只 spill，不每步读）

---

## 1. 三组对照

| 对照 | Key/Value 存在哪 | Decode 算哪些 | 配置 |
| --- | --- | --- | --- |
| **all_gpu_full** | 全部在 GPU | 完整 S | 无 working-set |
| **hier_full** | GPU 30% + SSD 70% | 完整 S；70% 每步从 SSD 回读 | `hier_30_0_70_slc_4k.json` |
| **sparse_offload** | 同上，GPU 再留 sink | 只算 260；回读 = 选中 ∩ 冷集 = **∅** | `hier_30_0_70_slc_4k_sparse256.json` |

**能否体现 SSD 读取：** 能，但只在 `hier_full`。`sparse_offload` 的 window 已在 GPU 尾部，decode 不读盘；SSD 只承担写出和驻留。

```bash
python3.11 gqa_working_set_test_report/s4096_sparse/run_sweep.py
```

```bash
python3.11 benchmark.py --batching paged-attn --qps 0.08 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b-gqa-4k.json \
  --verbose none --results_path gqa_working_set_test_report/s4096_sparse/all_gpu

python3.11 benchmark.py --batching paged-attn --qps 0.02 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b-gqa-4k.json \
  --verbose none --results_path gqa_working_set_test_report/s4096_sparse/hier_full \
  --kv_working_set_config data/kv_working_set/hier_30_0_70_slc_4k.json

python3.11 benchmark.py --batching paged-attn --qps 0.14 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b-gqa-4k.json \
  --verbose none --results_path gqa_working_set_test_report/s4096_sparse/sparse_offload \
  --kv_working_set_config data/kv_working_set/hier_30_0_70_slc_4k_sparse256.json
```

---

## 2. Decode 计算：attention 变稀疏，前馈网络不变

| 口径 | 上下文 S | Roofline (prompt, step) | attention 原始 | projection 原始 | 一步合计（校准后） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量，decode 开始 | ≈2049 | 2048+1 | 0.127 ms | 32.55 ms | 59.42 ms |
| 全量，decode 结束 | 4096 | 2048+2048 | **0.254 ms** | 32.55 ms | 59.65 ms |
| 稀疏，只算 260 | 逻辑 S=4096 | 259+1 | **0.016 ms** | 32.55 ms | 59.22 ms |

一步 decode 只少约 **0.7%**（projection 主导）。30/0/70 不改这张表，只改占用和回读。

---

## 3. 膝点

| 指标 | all_gpu_full | hier_full | sparse_offload |
| --- | ---: | ---: | ---: |
| **膝点到达率 λ\*** | **0.08** | **0.02**（轻载下限；goodput 已破） | **0.14** |
| **Little 并发 N\*** | 10.61 | 103.96 | 17.91 |
| 有效吞吐比 | 0.98 | **0.52** | 0.92 |
| 峰值并发（decode） | **16** | **53** | **53** |
| 峰值并发（prefill） | 32 | 32 | **32** |
| 存 / 算的 KV token | 4096 / 4096 | 4096 / 4096 | **4096 / 260** |
| 系统 decode token/s | **321.0** | **42.5** | **525.9** |
| 实际完成 r/s | 0.0784 | 0.0104 | 0.1284 |
| 首 token 时间 p50 | 0.288 s | 0.769 s | 0.311 s |
| 首 token 时间 p99 | 0.322 s | 2.65 s | 0.462 s |
| 每输出 token 时间 p50 | **64.6 ms** | **2545 ms** | **62.3 ms** |
| 每输出 token 时间 p99 | 66.5 ms | 2821 ms | 63.5 ms |
| 抢占次数 | 0 | 0 | 0 |
| SSD 读 token 数 | 0 | **4.404×10⁸** | **0** |
| 回读合计 / 写出合计 | 0 / 0 | **9600 s / 3.13 s** | **0 / 3.12 s** |

![膝点的每输出 token 时间、首 token 时间、系统 token/s](gqa_working_set_test_report/s4096_sparse/fig_knee_ttft_tpot_tokens.png)

![系统 token/s、每输出 token 时间、首 token 时间随到达率变化](gqa_working_set_test_report/s4096_sparse/fig_qps_ttft_tpot_tokens.png)

### 3.1 all_gpu_full vs hier_full — SSD 读取在这里

`hier_full` 的 token/s 是 all_gpu 的 **0.13×**。TPOT 从 64.6 ms 升到 **2.55 s**。70% 冷集每步从 SSD 回读：读 4.404×10⁸ token，回读合计 9600 s，DRAM 读 **0**。这就是「能体现出 SSD 读取」的臂。

λ=0.02 时 goodput 只有 **0.52**，已经不稳。λ=0.04 时 TTFT p99 升到 **4918 s**（10 次抢占）。搜索仍把 0.02 记为下限。

相对旧的 30/50/20（当时轻载还能 goodput 0.97、TPOT 179 ms）：去掉 DDR 之后，原先走 50 GB/s DRAM 的那一半改走 14 GB/s SSD，全量分层从「能跑」变成「轻载就饱和」。

### 3.2 all_gpu_full vs sparse_offload

`sparse_offload` 的 token/s 是 all_gpu 的 **1.64×**，是 hier_full 的 **12.4×**。中间 KV 写在 SSD 上（spill 3.12 s），decode **不读盘**。

同一到达率 **0.08**：

| 指标 | all_gpu_full | sparse_offload |
| --- | ---: | ---: |
| 系统 decode token/s | 321.0 | **322.4** |
| 每输出 token 时间 p50 | 64.59 ms | **60.87 ms** |
| 首 token 时间 p99 | 0.322 s | 0.357 s |
| SSD 读 token | 0 | **0** |

轻载 0.02：TPOT **59.69 vs 60.76 ms**。膝点 0.08 → 0.14 仍是占用（prefill 峰值 32），不是 SSD。λ=0.16 时 goodput 落到 0.90 以下（0.143/0.16）。

**结论：** 30/0/70 能展示 SSD **容量**（两臂都约 46 GiB）和全量臂的 SSD **读取**。稀疏臂在 `gpu_frac=0.30` 下读不到盘；要让稀疏 decode 也读 SSD，需要 window 伸进 70% 冷集（更小的 `gpu_frac`，或从中间选页）。

---

## 4. hier_full 的 SSD 读延迟

固定 30/0/70、4K、14 GB/s、`qd_cap=64`，到达率 **0.02**。不扫 `sparse_offload`（decode 无 SSD 读）。

13 µs 时 IOPS 上限由带宽封顶：14 GB/s ÷ 4096 B ≈ **3.42M**。

![hier_full 延迟扫描](gqa_working_set_test_report/s4096_sparse/fig_latency_ttft_tpot_tokens.png)

| SSD 读延迟 (µs) | 重叠方式 | IOPS 上限 | 系统 token/s | TPOT p50 | TTFT p50 | TTFT p99 | 回读合计 | goodput |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 层间预取 | 3.42M | **42.5** | **2545 ms** | 0.769 s | 2.65 s | 9600 s | 0.52 |
| 13 | 阻塞 | 3.42M | 40.2 | 2774 ms | 1.02 s | 2505 s | 9600 s | 0.49 |
| 25 | 层间预取 | 2.56M | 29.7 | 4222 ms | 1.38 s | 5843 s | 13762 s | 0.36 |
| 25 | 阻塞 | 2.56M | 28.7 | 4293 ms | 1.37 s | 6051 s | 13762 s | 0.35 |
| 50 | 层间预取 | 1.28M | 14.9 | 8892 ms | 2.61 s | 15516 s | 27525 s | 0.18 |
| 50 | 阻塞 | 1.28M | 14.6 | 9001 ms | 2.39 s | 15737 s | 27525 s | 0.18 |
| 100 | 层间预取 | 0.64M | 7.4 | 18404 ms | 5.32 s | 34421 s | 55050 s | 0.09 |
| 100 | 阻塞 | 0.64M | 7.4 | 18495 ms | 4.49 s | 34582 s | 55050 s | 0.09 |

全部点的 SSD 读 token 都是 **4.404×10⁸**（同一负载、同一 70% 冷集）。延迟加大只拉长排队，不改变读量。层间预取相对阻塞的优势被 SSD 带宽淹没（13 µs 时 42.5 vs 40.2 token/s）。

---

## 5. gpu_frac 扫描（dram_frac=0）：稀疏 decode 何时读盘

固定：`sparse=true`，sink=4，window=256，**dram_frac=0**（冷集全在 SSD），`ssd_frac = 1 - gpu_frac`。  
左图、表里的回读来自现有 `fetch_cost`（一步、单请求）。右图是同一套函数给出的该步回读延迟。虚线是阈值：`gpu_frac < window/S`。

```text
decode 开始 S=2048：gpu_frac < 256/2048 = 12.5% 才读盘
decode 结束 S=4096：gpu_frac < 256/4096 =  6.25% 才读盘
```

![fetch_cost：每步 SSD token 与回读延迟随 gpu_frac](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_fetch.png)

横轴从 30% 减到 2%（和「越降越容易出尾」一致）。`gpu_frac=0.30` 两条线都是 0。过了 12.5%，S=2048 的 window 开始伸出 GPU；过了 6.25%，结束时也开始读盘。多出来的 token 数 ≈ `256 - floor(S × gpu_frac)`，延迟大约每 50 个 token 1 ms 量级。

| gpu_frac | S=2048 尾部 / 每步 SSD 读 / ms | S=4096 尾部 / 每步 SSD 读 / ms | decode Peak B |
| ---: | --- | --- | ---: |
| 0.30 | 614 / **0** / 0 | 1228 / **0** / 0 | 53 |
| 0.15 | 307 / **0** / 0 | 614 / **0** / 0 | 105 |
| 0.12 | 245 / **11** / 0.24 | 491 / **0** / 0 | 132 |
| 0.10 | 204 / **52** / 1.13 | 409 / **0** / 0 | 157 |
| 0.08 | 163 / **93** / 2.03 | 327 / **0** / 0 | 195 |
| **0.06** | 122 / **134** / 2.92 | 245 / **11** / 0.24 | 256 |
| **0.05** | 102 / **154** / 3.36 | 204 / **52** / 1.13 | 315 |
| 0.04 | 81 / **175** / 3.81 | 163 / **93** / 2.03 | 373 |

0.12–0.07：只有 decode **前半段**读盘。**0.06 及以下**全程都读。

推理速度用 **每请求 decode token/s = 1 / TPOT**（不受到达率钉住）。右图是 Roofline 一步 59.22 ms 加上该步 `fetch_cost`：`1000 / (59.22 + fetch_ms)`。

![每请求 decode token/s：实测 1/TPOT，以及 Roofline+fetch_cost](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_speed.png)

| gpu_frac | 1/TPOT @0.08 | 1/TPOT @0.14 | 系统 token/s @0.08 | 系统 token/s @0.14 | TPOT @0.14 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | **16.43** | **16.04** | 322.4 | **525.9** | 62.35 ms |
| 0.15 | 16.42 | 16.02 | 322.4 | 525.8 | 62.41 ms |
| 0.12 | 16.42 | 16.02 | 322.4 | 525.8 | 62.42 ms |
| 0.10 | 16.41 | 16.01 | 322.4 | 525.8 | 62.47 ms |
| 0.08 | 16.39 | 15.98 | 322.4 | 525.8 | 62.60 ms |
| 0.06 | 16.35 | 15.91 | 322.3 | 525.6 | 62.86 ms |
| 0.05 | 16.33 | 15.84 | 322.3 | 525.4 | 63.13 ms |
| 0.04 | **16.30** | **13.65** | 322.2 | **516.4** | **73.24 ms** |

- 轻载 0.08：单请求速度 16.43 → 16.30 token/s（**−0.8%**），系统 token/s 几乎不动（到达率钉住）。  
- 膝点附近 0.14：0.30–0.05 仍约 16.0 → 15.8；**0.04** 掉到 **13.65** token/s，系统 **525.9 → 516.4**，TPOT 62.4 → **73.2 ms**，goodput 贴到 0.90。  
- Roofline+fetch：过 12.5% / 6.25% 之后一步速度从 16.9 往下走，和左图同一方向，幅度小于 0.14×0.04 的排队放大。

系统 token/s 与累计 SSD 读（对照）：

![系统 token/s、TPOT、累计 SSD 读](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_tokens.png)

和 `hier_full`（30/0/70、每步读约 2868 个 token、TPOT 2.55 s、token/s 42.5）比：稀疏打到盘时，读的仍是 window 多出来的几十到一百多个 token；只有到达率高且 `gpu_frac` 很低（0.04）时，推理速度才会明显掉。

复现：

```bash
python3.11 gqa_working_set_test_report/s4096_sparse/plot_gpu_frac_sparse.py
```

---

## 6. 实现

- 配置：[`hier_30_0_70_slc_4k.json`](data/kv_working_set/hier_30_0_70_slc_4k.json)、[`hier_30_0_70_slc_4k_sparse256.json`](data/kv_working_set/hier_30_0_70_slc_4k_sparse256.json)  
- `sparse=true`：存完整 KV；decode 只 attend / 只 fetch sink ∪ window  
- 膝点配方 fetch = 0（`gpu_frac=0.30`）；spill 全部写 SSD  
- `gpu_frac < window/S` 时稀疏 decode 开始读 SSD，见第 5 节  

未做：质量评测；Quest 式从中间选页（选中集在冷层，读量会大于 window 伸出的那一截）。
