# GQA 末端上下文 4096：稀疏 attention + KV offload（30/0/70）

Date / 日期: 2026-09-05  
Simulator / 模拟器: TokenSim（Roofline 后端，Python 3.11）  
Script / 脚本: [`gqa_working_set_test_report/s4096_sparse/run_sweep.py`](gqa_working_set_test_report/s4096_sparse/run_sweep.py)、[`plot_gpu_frac_sparse.py`](gqa_working_set_test_report/s4096_sparse/plot_gpu_frac_sparse.py)

本文独立，不替换 [`gqa_working_set_test_report.md`](gqa_working_set_test_report.md) / [`offload_sim.md`](offload_sim.md)。

分层改为 **GPU / DRAM / SSD = 30% / 0% / 70%**：冷集全部在 SSD，主机内存不存 KV。用来看 DDR、SSD 的极限容量，以及全量 attention 每步 SSD 回读。

**稀疏注意力可以和 offload 一起用。** 两条稀疏臂都把完整 Key/Value 留下（中间不淘汰），decode **只回读、只计算** 选中集：

- `sparse_offload`：选中集 = sink ∪ **滑动窗口** 256。window 已在 GPU 30% 尾部里，decode **回读为 0**。  
- `select_offload`：选中集 = sink ∪ **按重要性选出的 256 个 token**（Quest 式 top-k）。选中的页大多落在 70% 冷集里，decode **每步从 SSD 回读**。  
- `cache_offload`：同 `select_offload`，再在 GPU 上给每条请求留 **512 token 的页缓存**，装最近选中过的冷页；命中的不再读盘。  

**KV 选中策略：方案 A 和方案 C 都做了，都是统计模型，没有真实 trace。** TokenSim 没有真实 Q/K，算不出注意力分数，仓库里也没有任何选页 trace，所以：

- **方案 A**（`select_offload`）：top-k 建模成「在非 sink 上下文里均匀选 k 个 token」，每步回读期望 = `k × 冷集 / (S − sink)`。这是**最悲观**的命中假设。  
- **方案 C**（`cache_offload`）：在 A 上加 GPU 页缓存 `select_cache_tokens` 和相邻步重选比例 `select_reuse`。冷页命中率 = `reuse + (1 − reuse) × min(1, 缓存 / 冷集)`。主臂用 `reuse = 0`（不假设任何局部性，只算缓存容量的效果）；`reuse` 是 Quest 观察到的相邻 query 选页高度重合现象，没有 trace 就只能当参数扫（§3.4）。  

细节见 §6.1。

**先说增益从哪来。** `sparse_offload` 的膝点 0.14 vs `all_gpu_full` 0.08，**全部来自 GPU KV 占用下降带来的并发上限**（decode 峰值并发 16 → 53，被 prefill 的 32 卡住），不是 attention 少算 16× 带来的：decode 一步只快约 0.7%（§2）；同一到达率 0.08 两臂系统 token/s 相同（322 vs 321）。

**质量口径。** `all_gpu_full` 是全量 attention 的**性能**上限，不是质量分数。TokenSim 不生成文本，不算 perplexity / 准确率。两条稀疏臂的中间 KV 存在 SSD 上但 **不进本步 softmax**：`sparse_offload` 每步只看头 4 + 最近 256，输出与 `all_gpu_full` 不同，本质是 StreamingLLM 计算 + 全量存储；`select_offload` 能触达中间 token，文献里（Quest，k=256–1024）接近无损，但本文没有质量数字。

核心问题（性能）：

> 冷集全部放 SSD、不用 DDR 时，稀疏 decode 能否避开全量分层的每步 SSD 回读、提高可稳定的每秒请求数；如果选中集要从 SSD 里选页，代价是多少？

六条结论分开写：

1. **all_gpu_full vs hier_full** — 全量 attention 每步从 SSD 回读 70% 冷集的代价  
2. **all_gpu_full vs sparse_offload** — 滑窗稀疏 + offload 相对全量 GPU 的性能差  
3. **sparse_offload vs select_offload** — 从 SSD 里按重要性选页的回读代价（方案 A）  
4. **select_offload vs cache_offload** — GPU 页缓存能追回多少（方案 C，缓存 × 重选比例网格）  
5. **hier_full 的 SSD 读延迟扫描** — 只扫全量分层  
6. **gpu_frac 扫描（dram_frac=0）** — 滑窗、选页何时打到盘，以及 token/s 怎么变  

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
| window | 滑动窗口 | **256** 个，最新 token。`sparse_offload` 的选中集。 |
| select / top-k | 按重要性选页 | **256** 个，Quest 式按分数选出的 token。`select_offload` / `cache_offload` 的选中集；本文用均匀分布建模。 |
| 页缓存 / `select_cache_tokens` | GPU 上的冷页缓存 | `cache_offload` 每请求 **512** token（32 页），装最近选中过的冷页；占 GPU 块。 |
| `select_reuse` | 相邻步重选比例 | 本步选中集里有多大比例在上一步也被选中。主臂 **0**；§3.4 扫 0 / 0.5 / 0.8。 |
| HBM | GPU 高带宽显存 | 算力与 GPU 驻留 KV。 |
| DRAM / DDR | 主机内存 | 本实验 **0%**：不存 KV。 |
| SSD | 固态盘 | 冷集 **70%**。`hier_full` 每步读全部冷集；`select_offload` 每步读选中 ∩ 冷集；`sparse_offload` 只存、本配方 decode 不读。 |
| SLC | 盘档延迟 | 13 µs、14 GB/s。 |
| 30/0/70 | GPU / DRAM / SSD | 最新 30% 在 GPU，其余 70% 在 SSD。`sparse_offload` 的 GPU 再留 sink。 |
| 冷集 | 不在 GPU 上的 KV | 本实验 = 全部 SSD。 |
| fetch / spill | 回读 / 写出 | decode 读盘；prefill 结束后把洞写到 SSD。 |
| 4K / `qd_cap` | 命令大小 / 队列深度 | 4096 字节，上限 64。 |

### 0.2.1 gpu_frac / dram_frac / ssd_frac 是什么关系

三个数是 **每条请求当前上下文 S 的切分比例**，不是设备容量：

```text
gpu_frac + dram_frac + ssd_frac = 1                （配置校验强制）
GPU  = floor(S × gpu_frac)  最新的那一段         （sparse 时再加 sink 4 个）
DRAM = floor(S × dram_frac) 紧挨 GPU 段的更旧一段
SSD  = S − GPU − DRAM       最旧的一段（余数全归 SSD）
```

按「新 → 旧」排布：`[sink] [SSD 最旧 ... ] [DRAM ... ] [GPU 最新 ... S)`。S 每生成一个 token 加 1，三段随之重算，所以比例固定、token 数随 S 涨。

本文固定 `dram_frac = 0`，于是 **`ssd_frac = 1 − gpu_frac`**：第 5 节把 `gpu_frac` 从 30% 降到 4%，就是把 SSD 上的那段从 70% 放大到 96%，DRAM 始终为 0。S=4096 的例子：

| gpu_frac | GPU token（+sink） | DRAM token | SSD token | GPU MiB / 请求 | SSD MiB / 请求 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00（all_gpu_full） | 4096 | 0 | 0 | 1280 | 0 |
| 0.30 | 1228（1232） | 0 | 2868（2864） | 385 | 895 |
| 0.15 | 614（618） | 0 | 3478 | 193 | 1087 |
| 0.04 | 163（167） | 0 | 3929 | 52 | 1228 |

三条推论：

- **设备容量** = 该段 token × 0.3125 MiB × 同时在跑的请求数，见 §0.7。比例本身不是 GB。  
- **谁被读回来** 由 attention 策略决定，不由比例决定：全量 attention 每步读 DRAM+SSD 全部；滑窗只读伸出 GPU 的那截；选页读 `top-k ∩ (DRAM ∪ SSD)`。  
- **方案 C 的页缓存** 是 GPU 段之外的**额外**驻留（`select_cache_tokens`），不改三段切分，只多占 GPU 块、少读冷集。  

旧的 30/50/20 就是 `dram_frac=0.5`：S=4096 时 DRAM 2048 token、SSD 816 token。

### 0.3 并行、占用

| 简称 | 含义 |
| --- | --- |
| TP/PP/DP | 均为 **1**（单卡）。 |
| `paged-attn` | 每块 **16** 个 token。 |
| Peak B | `floor(可用 GPU 块数 / 每请求占用块数)`。 |
| watermark | 4144 块预留 1% → 可用 **4103**。 |

### 0.4 指标

TTFT = 首 token 时间（含 prefill）。TPOT = 每输出 token 时间（不含 prefill）。λ* = 膝点到达率（goodput ≥ 0.90 且 TTFT p99 ≤ 3 × 轻载 TTFT p99）。轻载 λ=0.02。`hier_full` 在 λ=0.02 时 goodput 已是 0.52，**没有膝点**；下文标为「未找到（≤0.02）」，其 Little N\* 不列（排队请求占多数，与其他臂不可比）。

膝点搜索的到达率网格是 0.02 → 0.04 → 0.08 → 0.16 再二分，所以 `all_gpu_full` 原本没有 0.14 的点；本文为它补跑了 0.14，曲线图上 0.14 处各臂都是实测。

### 0.5 五组对照

| 简称 | 做什么 | 与 all_gpu_full 输出是否相同 | 是否评质量 |
| --- | --- | --- | --- |
| all_gpu_full | 全量 KV 在 GPU + 完整上下文 attention | 是（基准） | 否 |
| hier_full | 30/0/70 + 完整上下文 attention；70% 每步从 SSD 回读 | 是 | 否 |
| sparse_offload | 30/0/70 存完整 KV；decode 只算 sink+window | **否**：中间 token 不进 softmax | 否 |
| select_offload | 30/0/70 存完整 KV；decode 只算 sink+top-k（均匀建模，方案 A） | **否**：只算 260 个；但能触达中间 | 否 |
| cache_offload | 同上 + GPU 512 token 页缓存（方案 C，reuse=0） | 同 select_offload（缓存不改选中集） | 否 |

### 0.6 实验配方

| 项 | 值 |
| --- | --- |
| 集群 | 1×H200 |
| 模型 | `LLaMa2-70B-GQA`，FP16 KV **0.3125 MiB/token** |
| 负载 | 100 条，prefill **2048±32**，decode = 该请求 prefill，`random_seed=0` |
| 实测 | 204,826 prefill + 204,826 decode |
| 选中集 | sink **4** + window **256**（sparse_offload）/ sink **4** + top-k **256**（select_offload、cache_offload） |
| 页缓存 | cache_offload：**512** token / 请求，`select_reuse=0` |
| Offload | **30/0/70** |

```text
全量 attention 峰值并发（S=4096）                     = floor(4103 / 256) = 16
hier_full / sparse_offload / select_offload（30% 尾部）= floor(4103 / 77)  = 53
cache_offload（30% 尾部 + 512 缓存 = 1744 token）     = floor(4103 / 109) = 37
Prefill 峰值并发（prompt ≈2048）                      = floor(4103 / 128) = 32
```

---

## 0.7 DDR / SSD 极限容量

口径：结束时 S=4096，每 token **0.3125 MiB**（整网 80 层）。一条请求完整 KV = **1280 MiB**。  
**极限驻留** = 该层每请求 token × decode 峰值并发 53（GPU 块刚好能塞满 53 条 decode）。这是「要同时撑住 Peak B 条满上下文请求」时，DDR/SSD 至少要备的容量，不是仿真过程中累计读写流量。**53 是理论墙**：本负载 prefill 墙 32 先卡住，膝点处 Little N\* 只有 14–18，53 条满上下文 decode 同时在跑的情形没出现过；按实测并发折算的驻留见最后一行。

| | all_gpu_full | hier_full | sparse_offload | select_offload | cache_offload |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU token / 请求 | 4096 | 1228 | 1232（sink+尾部） | 1232 | **1744**（+512 缓存） |
| DDR token / 请求 | 0 | **0** | **0** | **0** | **0** |
| SSD token / 请求 | 0 | **2868** | **2864** | **2864** | **2864**（缓存是副本，洞仍全写盘） |
| GPU MiB / 请求 | 1280 | 383.8 | 385.0 | 385.0 | 545.0 |
| **DDR MiB / 请求** | 0 | **0** | **0** | **0** | **0** |
| **SSD MiB / 请求** | 0 | **896.3** | **895.0** | **895.0** | **895.0** |
| GPU 峰值并发（理论） | 16 | 53 | 53 | 53 | **37** |
| **DDR 极限** | 0 | **0** | **0** | **0** | **0** |
| **SSD 极限（×Peak B）** | 0 | **46.4 GiB** | **46.3 GiB** | **46.3 GiB** | **32.3 GiB** |
| 膝点实测 Little N\* | 10.6 | —（未找到膝点） | 17.9 | 14.2 | 15.8 |
| SSD 按实测 N\* 折算 | 0 | — | **15.6 GiB** | **12.4 GiB** | **13.8 GiB** |

对照：若仍用旧的 30/50/20，同样 Peak B=53 时 DDR 极限约 **33.1 GiB**、SSD 约 **13.3 GiB**。30/0/70 把那 33 GiB DDR 全部改到盘上，DDR 极限为 0，SSD 极限约 **46 GiB**。

Prefill 峰值 32 条、每条约 2048 token 全在 GPU：32 × 2048 × 0.3125 = **20.0 GiB**，与可用 4103 块（约 20.0 GiB）对齐。

仿真累计 I/O（100 条请求、不是驻留容量）：

- `hier_full`：SSD **读** 4.404×10⁸ token，回读合计 **9600 s**；写出 47.0 GB  
- `sparse_offload`：SSD **读 0**；写出 46.9 GB（只 spill，不每步读）  
- `select_offload`：SSD **读** 3.665×10⁷ token（hier_full 的 **1/12**），回读合计 **799 s**；写出 46.9 GB  
- `cache_offload`：SSD **读** 2.758×10⁷ token（select_offload 的 **0.75×**），回读合计 **601 s**；写出 46.9 GB

---

## 1. 五组对照

| 对照 | Key/Value 存在哪 | Decode 算哪些 | 配置 |
| --- | --- | --- | --- |
| **all_gpu_full** | 全部在 GPU | 完整 S | 无 working-set |
| **hier_full** | GPU 30% + SSD 70% | 完整 S；70% 每步从 SSD 回读 | `hier_30_0_70_slc_4k.json` |
| **sparse_offload** | 同上，GPU 再留 sink | 只算 260；回读 = 窗口 ∩ 冷集 = **∅** | `hier_30_0_70_slc_4k_sparse256.json` |
| **select_offload** | 同上 | 只算 260；回读 = top-k ∩ 冷集 ≈ **179 token/步**（均匀） | `hier_30_0_70_slc_4k_select256.json` |
| **cache_offload** | 同上 + GPU 512 token 页缓存 | 只算 260；回读 = 179 × (1 − 512/2864) ≈ **147 token/步** | `hier_30_0_70_slc_4k_cache512.json` |

**能否体现 SSD 读取：** 能。`hier_full` 每步读整个冷集；`select_offload` 每步读选中集里落在冷集的那部分（S=4096 时 256 × 2864/4092 ≈ 179 个 token ≈ 56 MiB）；`cache_offload` 再扣掉缓存命中；`sparse_offload` 的 window 已在 GPU 尾部，decode 不读盘，SSD 只承担写出和驻留。

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

python3.11 benchmark.py --batching paged-attn --qps 0.1 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b-gqa-4k.json \
  --verbose none --results_path gqa_working_set_test_report/s4096_sparse/select_offload \
  --kv_working_set_config data/kv_working_set/hier_30_0_70_slc_4k_select256.json

python3.11 benchmark.py --batching paged-attn --qps 0.12 --distribution poisson \
  --cluster data/clusters/1_h200/h1.json --model data/psla/llama-70b-gqa-4k.json \
  --verbose none --results_path gqa_working_set_test_report/s4096_sparse/cache_offload \
  --kv_working_set_config data/kv_working_set/hier_30_0_70_slc_4k_cache512.json
```

---

## 2. Decode 计算：attention 变稀疏，前馈网络不变

| 口径 | 上下文 S | Roofline (prompt, step) | attention 原始 | projection 原始 | 一步合计（校准后） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量，decode 开始 | ≈2049 | 2048+1 | 0.127 ms | 32.55 ms | 59.42 ms |
| 全量，decode 结束 | 4096 | 2048+2048 | **0.254 ms** | 32.55 ms | 59.65 ms |
| 稀疏，只算 260 | 逻辑 S=4096 | 259+1 | **0.016 ms** | 32.55 ms | 59.22 ms |

一步 decode 只少约 **0.7%**（projection 主导）。30/0/70 不改这张表，只改占用和回读。`select_offload` 的 attention 行与「稀疏，只算 260」相同；Quest 的页元数据打分（每页一次小点积）未计入，量级远小于 projection。

---

## 3. 膝点

| 指标 | all_gpu_full | hier_full | sparse_offload | select_offload | cache_offload |
| --- | ---: | ---: | ---: | ---: | ---: |
| **膝点到达率 λ\*** | **0.08** | **未找到（≤0.02）** | **0.14** | **0.10** | **0.12** |
| **Little 并发 N\*** | 10.61 | — | 17.91 | 14.21 | 15.79 |
| 有效吞吐比 | 0.98 | **0.52**（λ=0.02） | 0.92 | 0.95 | 0.93 |
| 峰值并发（decode，理论） | **16** | **53** | **53** | **53** | **37** |
| 峰值并发（prefill） | 32 | 32 | **32** | 32 | 32 |
| 存 / 算的 KV token | 4096 / 4096 | 4096 / 4096 | **4096 / 260** | **4096 / 260** | **4096 / 260** |
| 系统 decode token/s | **321.0** | **42.5** | **525.9** | **389.6** | **459.0** |
| 实际完成 r/s | 0.0784 | 0.0104 | 0.1284 | 0.0951 | 0.1120 |
| 首 token 时间 p50 | 0.288 s | 0.769 s | 0.311 s | 0.314 s | 0.311 s |
| 首 token 时间 p99 | 0.322 s | 2.65 s | 0.462 s | 0.426 s | 0.432 s |
| 每输出 token 时间 p50 | **64.6 ms** | **2545 ms** | **62.3 ms** | **69.5 ms** | **64.3 ms** |
| 每输出 token 时间 p99 | 66.5 ms | 2821 ms | 63.5 ms | 77.9 ms | 67.1 ms |
| 抢占次数 | 0 | 0 | 0 | 0 | 0 |
| SSD 读 token 数 | 0 | **4.404×10⁸** | **0** | **3.665×10⁷** | **2.758×10⁷** |
| 回读合计 / 写出合计 | 0 / 0 | **9600 s / 3.13 s** | **0 / 3.12 s** | **799 s / 3.12 s** | **601 s / 3.12 s** |

`hier_full` 一列是 λ=0.02 的实测值，不是膝点。

![膝点的每输出 token 时间、首 token 时间、系统 token/s](gqa_working_set_test_report/s4096_sparse/fig_knee_ttft_tpot_tokens.png)

![系统 token/s、每输出 token 时间、首 token 时间随到达率变化](gqa_working_set_test_report/s4096_sparse/fig_qps_ttft_tpot_tokens.png)

曲线图怎么读：左图纵轴是**系统** token/s（100 条请求吐出的 token ÷ 墙钟），到达率越高分母越短，所以过膝之后曲线还会涨。`all_gpu_full` 在 0.14 实测 486.5 token/s，但 TTFT p99 已经 **64 s**、抢占 28 次——那是把请求挤在更短时间里的算术，不是能稳定接待 0.14。各臂在同一到达率下（如 0.08）系统 token/s 几乎一样；差别在**能稳到多高**。

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

**结论：** 30/0/70 能展示 SSD **容量**（两臂都约 46 GiB）和全量臂的 SSD **读取**。滑窗稀疏臂在 `gpu_frac=0.30` 下读不到盘；要让稀疏 decode 也读 SSD，需要 window 伸进 70% 冷集（更小的 `gpu_frac`，§5），或者像 §3.3 那样从中间选页。

### 3.3 sparse_offload vs select_offload — 从 SSD 里选页的代价

两臂存储、占用完全一样（GPU 1232 token、SSD 2864、Peak B 53），attention 都只算 260。唯一区别是选中集在哪：

- `sparse_offload`：window 全在 GPU 尾部，每步回读 **0**。  
- `select_offload`：均匀选 256 个，期望 **179** 个落在 SSD 冷集，每步每请求约 **56 MiB**、**14,320** 个 4K IO；14 GB/s 下约 3.9 ms/请求。

| 到达率 | sparse TPOT p50 | select TPOT p50 | select goodput | select TTFT p99 |
| ---: | ---: | ---: | ---: | ---: |
| 0.02 | 59.7 ms | 59.8 ms | 1.06 | 0.342 s |
| 0.08 | 60.9 ms | 61.8 ms | 0.98 | 0.350 s |
| **0.10** | — | **69.5 ms** | **0.95** | 0.426 s |
| 0.12 | 61.7 ms | 96.9 ms | **0.87** | 0.412 s |
| 0.14 | 62.3 ms | 130.8 ms | 0.79 | 0.467 s |
| 0.16 | 62.8 ms | 168.6 ms | 0.71 | 0.634 s |

- 轻载时 `layer_prefetch` 把 3.9 ms 的回读藏在 80 层逐层计算后面，TPOT 几乎不变。  
- 并发到 14 条以上，每步 SSD 总读量 ≈ 14 × 56 MiB ≈ 0.8 GB、约 55 ms，和 59 ms 的计算持平，重叠不住了；TPOT 从 0.10 起随并发线性上涨，0.12 goodput 掉到 0.87。  
- 膝点 **0.10** vs `sparse_offload` 0.14：选页回读把稀疏臂的增益从 1.64× 压到 **1.21×**（相对 all_gpu），但仍比 `hier_full`（每步读 2868 token）好一个数量级：累计 SSD 读 3.7×10⁷ vs 4.4×10⁸。  
- 这是均匀选择（方案 A）的悲观值。若真实选页有近因偏置（更多命中 GPU 尾部），或者加页缓存（方案 C，§3.4），回读会更少。

### 3.4 select_offload vs cache_offload — 方案 C 能追回多少

`cache_offload` 在 `select_offload` 之上给每条 decode 请求留 **512 token（32 页）** 的 GPU 页缓存，装最近选中过的冷页。代价是每请求 GPU 占用 1232 → **1744** token（77 → 109 块），decode 理论并发 53 → **37**；prefill 墙 32 仍先卡住，所以对本负载没有实际损失。

主臂 `select_reuse = 0`：不假设相邻步有任何重合，命中率只来自「缓存装了冷集的多大比例」= 512/2864 = **17.9%**。每步回读 179 → **147** token。

| | select_offload | cache_offload（512, reuse 0） |
| --- | ---: | ---: |
| 膝点 λ\* | 0.10 | **0.12** |
| 系统 token/s | 389.6 | **459.0** |
| TPOT p50 @膝点 | 69.5 ms | 64.3 ms |
| TPOT p50 @0.14 | 130.8 ms | 72.4 ms |
| goodput @0.14 | 0.79 | **0.895**（差 0.005 没过 0.90） |
| 累计 SSD 读 | 3.665×10⁷ | 2.758×10⁷（**−25%**） |

**缓存 × 重选比例网格**（都是膝点搜索，`reuse` 是假设值，无 trace）：

| 缓存 token | reuse | 每步冷读（S=4096） | λ\* | 系统 token/s | TPOT p50 | 累计 SSD 读 | Peak B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0（select_offload） | 0 | 179 | 0.10 | 389.6 | 69.5 ms | 3.665×10⁷ | 53 |
| 512 | 0 | 147 | 0.12 | 459.0 | 64.3 ms | 2.758×10⁷ | 37 |
| 512 | 0.5 | 74 | **0.14** | 525.5 | 62.7 ms | 1.379×10⁷ | 37 |
| 512 | 0.8 | 29 | **0.14** | 525.7 | 62.5 ms | 0.551×10⁷ | 37 |
| 1024 | 0 | 115 | **0.14** | 525.3 | 62.9 ms | 1.849×10⁷ | 29 |
| 1024 | 0.5 | 58 | **0.14** | 525.6 | 62.6 ms | 0.924×10⁷ | 29 |
| 1024 | 0.8 | 23 | **0.14** | 525.7 | 62.5 ms | 0.370×10⁷ | 29 |
| —（sparse_offload 滑窗） | — | 0 | **0.14** | 525.9 | 62.3 ms | 0 | 53 |

读法：

- 决定膝点的是「每步全体请求的 SSD 读能不能被 `layer_prefetch` 藏在 59 ms 的计算后面」。每步冷读 ≤ 约 115 token（≈36 MiB/请求，18 条并发 ≈ 45 ms）就能追平滑窗的 0.14；179 token 时 SSD 时间超过计算，膝点掉到 0.10。  
- **只靠容量、不假设局部性**（reuse 0）：512 token 缓存不够（0.12），1024 才够（0.14）。但 1024 缓存让 Peak B 掉到 **29**，低于 prefill 墙 32——decode 占用反过来成了瓶颈，只是本负载 N\* 18 还没碰到。  
- **有局部性**（reuse ≥ 0.5）：512 缓存就够到 0.14，累计 SSD 读只有 select_offload 的 1/3–1/7。Quest 报告相邻 query 的选页高度重合，方向上支持 reuse 不为 0，但具体数值没有 trace 不能定。  
- 网格里所有到 0.14 的配置，TPOT 都在 62.5–62.9 ms，和滑窗（62.3）几乎一样：一旦回读藏得住，选页与滑窗在性能上等价，差别只剩「是否触达中间 token」。

---

## 4. hier_full 的 SSD 读延迟

固定 30/0/70、4K、14 GB/s、`qd_cap=64`，到达率 **0.02**。不扫 `sparse_offload`（decode 无 SSD 读）。

IOPS 上限 = `min(qd_cap / L, BW / io_size)`。13 µs 时 `64 / 13 µs ≈ 4.9M` 高于 `14 GB/s ÷ 4096 B ≈ 3.42M`，由带宽封顶；25 µs 起 `qd_cap / L` 成为瓶颈（2.56M、1.28M、0.64M）。

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

## 5. gpu_frac 扫描（dram_frac=0）：两种选中策略何时读盘

固定：`sparse=true`，sink=4，**dram_frac=0**（冷集全在 SSD），`ssd_frac = 1 - gpu_frac`（三者关系见 §0.2.1）。两种选中集：**window256**（实线）和 **select256 均匀选页**（点线）；不含页缓存。  
左图、表里的回读来自现有 `fetch_cost`（一步、单请求）。右图是同一套函数给出的该步回读延迟。虚线是滑窗的阈值：`gpu_frac < window/S`。

```text
window：decode 开始 S=2048：gpu_frac < 256/2048 = 12.5% 才读盘
        decode 结束 S=4096：gpu_frac < 256/4096 =  6.25% 才读盘
select：任何 gpu_frac < 1 都读盘；每步期望 256 × (1 - gpu_frac) 左右
```

![fetch_cost：每步 SSD token 与回读延迟随 gpu_frac](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_fetch.png)

横轴从 30% 减到 2%（和「越降越容易出尾」一致）。滑窗在 `gpu_frac=0.30` 两条线都是 0；过了 12.5%，S=2048 的 window 开始伸出 GPU；过了 6.25%，结束时也开始读盘，多出来的 token 数 ≈ `256 - floor(S × gpu_frac)`。选页与 S 无关（比例式），30% 时每步 179 个，2% 时 251 个，接近全部 256。延迟大约每 50 个 token 1 ms 量级。

| gpu_frac | window S=2048 尾部 / SSD 读 / ms | window S=4096 尾部 / SSD 读 / ms | select S=2048 与 4096 SSD 读 / ms | decode Peak B |
| ---: | --- | --- | --- | ---: |
| 0.30 | 614 / **0** / 0 | 1228 / **0** / 0 | **179** / 3.90 | 53 |
| 0.15 | 307 / **0** / 0 | 614 / **0** / 0 | 218 / 4.75 | 105 |
| 0.12 | 245 / **11** / 0.24 | 491 / **0** / 0 | 225 / 4.90 | 132 |
| 0.10 | 204 / **52** / 1.13 | 409 / **0** / 0 | 230 / 5.01 | 157 |
| 0.08 | 163 / **93** / 2.03 | 327 / **0** / 0 | 236 / 5.14 | 195 |
| **0.06** | 122 / **134** / 2.92 | 245 / **11** / 0.24 | 241 / 5.25 | 256 |
| **0.05** | 102 / **154** / 3.36 | 204 / **52** / 1.13 | 243 / 5.30 | 315 |
| 0.04 | 81 / **175** / 3.81 | 163 / **93** / 2.03 | 246 / 5.36 | 373 |

滑窗 0.12–0.07：只有 decode **前半段**读盘；**0.06 及以下**全程都读。选页：全程都读。

推理速度用 **每请求 decode token/s = 1 / TPOT**（不受到达率钉住）。右图是 Roofline 一步 59.22 ms 加上该步 `fetch_cost`：`1000 / (59.22 + fetch_ms)`。

![每请求 decode token/s：实测 1/TPOT，以及 Roofline+fetch_cost](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_speed.png)

| gpu_frac | window 1/TPOT @0.08 | window 1/TPOT @0.14 | window token/s @0.14 | select 1/TPOT @0.08 | select 1/TPOT @0.14 | select token/s @0.14 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | **16.43** | **16.04** | **525.9** | **16.19** | **7.65** | **450.8** |
| 0.15 | 16.42 | 16.02 | 525.8 | 15.01 | 4.68 | 389.4 |
| 0.12 | 16.42 | 16.02 | 525.8 | 14.29 | 4.35 | 379.0 |
| 0.10 | 16.41 | 16.01 | 525.8 | 13.85 | 4.13 | 371.9 |
| 0.08 | 16.39 | 15.98 | 525.8 | 13.22 | 3.93 | 363.8 |
| 0.06 | 16.35 | 15.91 | 525.6 | 12.59 | 3.77 | 357.0 |
| 0.05 | 16.33 | 15.84 | 525.4 | 12.31 | 3.71 | 354.4 |
| 0.04 | **16.30** | **13.65** | **516.4** | **11.90** | **3.63** | **350.6** |

滑窗（window256）：

- 轻载 0.08：单请求速度 16.43 → 16.30 token/s（**−0.8%**），系统 token/s 几乎不动（到达率钉住）。  
- 膝点附近 0.14：0.30–0.05 仍约 16.0 → 15.8；**0.04** 掉到 **13.65** token/s，系统 **525.9 → 516.4**，TPOT 62.4 → **73.2 ms**，goodput 贴到 0.90。  
- Roofline+fetch：过 12.5% / 6.25% 之后一步速度从 16.9 往下走，和左图同一方向，幅度小于 0.14×0.04 的排队放大。

选页（select256，均匀）：

- 轻载 0.08：`gpu_frac` 从 0.30 降到 0.04，单请求 16.19 → 11.90 token/s（**−27%**），TPOT 61.8 → 84.0 ms；系统 token/s 322 → 310。每步读盘 179 → 246 个 token，`layer_prefetch` 在低并发下只能藏住一部分。  
- 0.14 已经过了选页臂的膝点（0.10）：TPOT 131 → 276 ms，单请求 7.65 → 3.63 token/s。这一列是「SSD 排队饱和」状态，不是稳态服务。  
- 降 `gpu_frac` 对选页几乎没有换来什么：GPU 驻留从 1232 降到 167 token，Peak B 从 53 升到 373，但 prefill 墙仍是 32，而每步读盘量还多了 37%。

系统 token/s 与累计 SSD 读（对照；右图只画到达率 0.08）：

![系统 token/s、TPOT、累计 SSD 读](gqa_working_set_test_report/s4096_sparse/fig_gpu_frac_tokens.png)

和 `hier_full`（30/0/70、每步读约 2868 个 token、TPOT 2.55 s、token/s 42.5）比：滑窗打到盘时，读的是 window 多出来的几十到一百多个 token；选页每步读 179–251 个，都比全量的 2868 少一个数量级，但选页是每步都读，累计 SSD 读 3.7–5.0×10⁷ token，滑窗最多 2.7×10⁷（0.04）。

复现：

```bash
python3.11 gqa_working_set_test_report/s4096_sparse/plot_gpu_frac_sparse.py
```

---

## 6. 实现

- 配置：[`hier_30_0_70_slc_4k.json`](data/kv_working_set/hier_30_0_70_slc_4k.json)、[`hier_30_0_70_slc_4k_sparse256.json`](data/kv_working_set/hier_30_0_70_slc_4k_sparse256.json)、[`hier_30_0_70_slc_4k_select256.json`](data/kv_working_set/hier_30_0_70_slc_4k_select256.json)、[`hier_30_0_70_slc_4k_cache512.json`](data/kv_working_set/hier_30_0_70_slc_4k_cache512.json)  
- `sparse=true`：存完整 KV；decode 只 attend / 只 fetch 选中集  
  - `select_tokens=0`（默认）：选中集 = sink ∪ window；fetch = window ∩ 冷集  
  - `select_tokens=k>0`：选中集 = sink ∪ top-k；fetch = `k × 冷集 / (S − sink)` 的期望值，按 DRAM/SSD 在冷集中的比例拆分（`TokenSim/kv_working_set/fetch.py::_selected_cold_tokens`）  
  - `select_cache_tokens=C`、`select_reuse=p`：冷页命中率 `p + (1 − p) × min(1, C / 冷集)`，fetch 乘 `(1 − 命中率)`；C 计入每请求 GPU 块（`placement.py::gpu_resident_tokens`、`block_manager._gpu_target_blocks`），要求 `C ≥ k`  
- 膝点配方：`sparse_offload` fetch = 0（`gpu_frac=0.30`）；`select_offload` 每步 179 token；`cache_offload` 每步 147 token；spill 全部写 SSD  
- `gpu_frac < window/S` 时滑窗 decode 开始读 SSD，见第 5 节。第 5 节没有扫页缓存。  
- 方案 C 网格：`run_sweep.py::run_cache_grid` → `cache_grid.json`

### 6.1 KV 选中策略：方案 A 与方案 C，都是统计模型

| | 方案 A `select_offload` | 方案 C `cache_offload` |
| --- | --- | --- |
| 选中哪些页 | 均匀分布在非 sink 上下文上；每步算期望值 | 同 A |
| GPU 上放什么 | sink + 最新 30% 尾部 | 同 A + `select_cache_tokens` 页缓存（装最近选中过的冷页） |
| 每步回读 | `k × 冷集 / (S − sink)`，S=4096 时 179 | 再乘 `(1 − 命中率)`；命中率 = `reuse + (1 − reuse) × 缓存/冷集` |
| 状态 | 无；`fetch_cost` 是纯函数 | **仍然无**：均匀选择下任何 demand-paging 缓存的命中率恰好等于「缓存 / 冷集」，相邻步重选部分必然在缓存里，所以闭式期望等于逐页 LRU 仿真的均值 |
| 假设 | 均匀（悲观） | 均匀 + `reuse`；`reuse` 无 trace，只能扫 |
| 主臂参数 | k=256 | k=256，C=512，reuse=0 |

**为什么 C 也不做逐页 LRU 状态机。** 没有真实 trace 时，选页只能是随机模型；在随机模型下逐页 LRU 只是给同一个期望值加噪声。`reuse` 参数已经把「相邻步高度重合」这一 Quest 观察到的结构表达出来了；要再往前走，缺的是 trace，不是状态机。

**C 相对 A 的结论**：容量本身（reuse=0）要到 1024 token 才追平滑窗，且把 decode 并发墙压到 29；局部性（reuse ≥ 0.5）让 512 就够。没有 trace 前，报告只把 reuse=0 作为主臂。

未做：质量评测（TokenSim 不算 perplexity / 准确率）；Quest 页元数据打分开销；基于真实 trace 的选页分布（方案 B）；页缓存的 `gpu_frac` 扫描。

### 6.2 结果目录里的历史遗留

`gqa_working_set_test_report/s4096_sparse/` 下的 `hier_sparse256/`、`streaming_sparse/`，以及 `data/kv_working_set/hier_30_50_20_*.json`，来自更早的实验设计（30/50/20 分层、StreamingLLM 淘汰），本文不引用。
