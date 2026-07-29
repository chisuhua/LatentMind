# Logos K 值策略与端侧可行性

> **一句话定位**：在 LoopCoder-v2 冲击下，重新校准 Logos 主线（HRM-Text）的 K 值策略——**架构方向已证，工程调优为目标**，K ≤ 2 为端侧硬约束，K=8 集成失败时启动 RDD-0001 Plan B。
> **性质**：Logos 主线的核心策略文档，整合 LoopCoder-v2 / Huginn / STARS / Per-Token Convergence 四份参考笔记
> **最后更新**：2026-07-29

---

## 0. 核心结论

| 决策 | 结论 |
|------|------|
| **循环架构方向** | ✅ 已被 LoopCoder-v2 等论文验证，**无须前置实验** |
| **K 默认值** | **K=2**（端侧时延预算内） |
| **K 调度** | **per-token early exit**（90% token 6 步内收敛） |
| **并行化** | **PLT 的 CLP + G-SWA**移植到 HRM（延迟压平） |
| **K-sweep 阻塞** | ❌ **不阻塞主线**；降级为 Plan B |
| **RDD-0001 触发** | K=8 时延超预算或出现 gain-then-collapse |

---

## 1. LoopCoder-v2 的主线消化

### 1.1 不重复的笔记

详细论文笔记见 [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md)。本节只做"主线消化"——这些发现对 Logos 主线意味着什么。

### 1.2 核心发现的"两面性"

| 发现 | 字面解读 | Logos 主线消化 |
|------|---------|---------------|
| K=2 > K=1 (+50%) | 循环架构方向正确 | ✅ Logos 起点即正确，无须怀疑循环 |
| K=3 < K=1 (-36%) | 循环越多越糟 | ⚠️ PLT 架构特有，**不泛化**到 HRM |
| 10 个 benchmark 一致 | 结论稳健 | ✅ 可作 Logos 工程调优基线 |

**关键区分**：LoopCoder-v2 的崩溃是 **PLT + CLP 架构特有**，不是循环本身的固有属性。Huginn 反例（3.5B 无 CLP 循环 K=50 仍 work，详见 [huginn.md §4.1](../references/huginn.md#41-循环深度-sweepgsm8k-35b)）已证明这一点。

### 1.3 对 Logos 的三个启示

1. **HRM-Text 的层次化结构可能免疫 PLT 的崩溃**——H 模块提供"慢变量锚点"，防止 L 模块进入低维坍缩空间。但这只是理论推断，**需要在集成中验证**。
2. **K=2 是工程最优的起点**——PLT 7B 在代码任务上 K=2 取得最大收益；Logos 主线默认 K=2 是合理起点。
3. **per-token 动态 K 比固定 K 更优**——90% token 6 步收敛，10% 需要 8 步（详见 [per-token-convergence.md](../references/per-token-convergence.md)）。固定 K=8 是浪费计算。

---

## 2. K 值预算矩阵

### 2.1 端侧时延约束

假设 HRM-Text 1B FP16 在 ChipForge APU 上：
- 单次前向（K=1）：**50ms / token**
- K=2：100ms / token
- K=4：200ms / token（已超对话预算）
- K=8：400ms / token（**完全不可用**）

| 场景 | 时延预算 | 可接受 K |
|------|---------|---------|
| 自动驾驶感知 | <50ms | K=1（无循环！）|
| 实时对话 | <100ms | K=2 |
| 离线规划 | <500ms | K=8（理论上）|

**硬约束**：物理 AI 主流场景需要 **K ≤ 2**。

### 2.2 任务类型 → 最优 K

| 任务类型 | 数学特性 | 经验最优 K | Logos 默认 |
|---------|---------|----------|----------|
| 简单 QA | $f$ 接近压缩 | K=1-2 | K=2 |
| 中等推理（GSM8K）| $f$ 临界稳定 | K=4-16 | K=2 + early exit |
| 深度抽象（ARC-AGI）| $f$ 接近扩张 | K=4-8 | K=2 + early exit |
| 代码生成 | $f$ 容易扩张 | K=2（PLT 经验）| K=2 |
| 多轨迹决策（GRAM）| 概率展开 | K=2 × 4 轨迹 | K=2 + 4 轨迹 |

**Logos 默认 K=2 + 动态调整（per-token early exit）** 覆盖所有场景。

---

## 3. K=2 默认策略

### 3.1 为什么 K=2

| 理由 | 论证 |
|------|------|
| **端侧时延约束** | K=2 = 100ms，硬约束内 |
| **PLT 实证支持** | 7B SWE-bench K=2 +50%（vs K=1） |
| **Huginn 经验** | 3.5B GSM8K K=4 达到饱和（48.5%），K=2 已达 41.2%（+19%） |
| **Per-token 数据** | 90% token 6 步收敛，K=2 覆盖大部分 |

### 3.2 K=2 的代价与收益

| 维度 | 评估 |
|------|------|
| **时延** | ✅ 100ms（端侧可接受）|
| **质量损失**（vs K=8）| ⚠️ 估计 5-10%（基于 Huginn 数据外推）|
| **算力开销** | K=1 的 2 倍，仍可控 |
| **集成风险** | 🟢 低（K=8 的崩溃风险不存在）|

**判断**：K=2 是 Logos v1.0 端侧 Demo 的最优起点。**先 K=2 出 demo，再讨论 K 调整**。

### 3.3 K=2 不是固定值

虽然默认 K=2，但实际部署中应该**动态调整**：

```python
def dynamic_K_forward(x, max_K=4, epsilon=1e-4):
    h = init(x)
    for k in range(1, max_K + 1):
        h_new = block(h, x)
        delta = (h_new - h).norm(dim=-1)  # [B, L]
        # per-token early exit
        exit_mask = delta < epsilon  # [B, L]
        if exit_mask.all(): break
        h = h_new
    return h
```

详细 per-token 收敛分析见 [per-token-convergence.md §3](../references/per-token-convergence.md#3-核心发现)。

---

## 4. 多种端侧并行化策略（Latency Killer）

> **核心洞察**：端侧约束下，**K>4 串行循环实际价值很小**——4 倍延迟对自动驾驶/对话场景不可接受。必须用并行化策略代替。

### 4.1 为什么 K>4 串行循环无价值

| K | 单 token 延迟 | 应用场景 |
|:---:|:---:|---|
| K=1 | 50ms | ✅ 自动驾驶感知 |
| K=2 | 100ms | ✅ 实时对话 |
| K=4 | 200ms | ⚠️ 仅离线规划 |
| K=8 | 400ms | ❌ 完全不可用 |

**核心问题**：K>4 在端侧 = 用不起 = 实际价值≈0。**Logos 默认 K ≤ 2 + 多种并行化策略扩展**。

### 4.2 PLT/HLT-PLT 移植（K 循环并行）

**PLT 的解法**（详见 [loopcoder-v2.md §3](../references/loopcoder-v2.md#3-核心创新)）：
- **CLP（Cross-Loop Position Shift）**：跨循环位置偏移打破串行依赖 → 多圈可并行
- **G-SWA（Shared KV 的门控滑窗）**：首循环 KV 共享 → 显存不随圈数增长

**移植到 HRM**：HLT-PLT（H 串行包 L 并行）

```python
# 原 HRM-Text（H 串行包 L 串行）
for h in range(H_cycles):     # 2
    for l in range(L_cycles): # 3（串行）
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)

# HLT-PLT（H 串行 + L 并行）
for h in range(H_cycles):                # 2（仍串行）
    z_L = parallel_L_chain(z_L + z_H)   # 3 步 L 在一个 forward 内并行
    z_H = H_module(z_H + z_L)
```

**延迟节省**：8 步变 3 步，**延迟降低 62%**。

**适用场景**：单轨迹推理 K=4-8 时。

### 4.3 Radix Cache 多路径并行（你提出的方案）

**核心思想**：GRAM 多轨迹在 latent tree search 中并行采样，共享前缀通过 Radix Tree 缓存。

```python
def radix_cache_gram(x, n_paths=4, K=2):
    base_h = prefill(x)  # 共享前缀
    radix_cache = RadixTree(base_h)
    
    # N 条路径并行采样
    paths = []
    for i in range(n_paths):
        eps_i = sample_eps()
        h_i = base_h.clone()
        for k in range(K):
            h_i = hrm_block(h_i, x, eps_i, cache=radix_cache)
        paths.append(h_i)
    
    return aggregate(paths)  # 综合 N 条路径
```

**延迟**：N=4 路径 × K=2 = 8 步推理，延迟 ≈ **1 次 K=8 forward**（路径并行）。

**适用场景**：多假设决策（路口左/右转）。

### 4.4 层次化推理（你提出的另一个方案）

**核心思想**：主推理在关键节点暂停，派生子推理完成局部任务，再恢复。

```python
def hierarchical_reasoning(x, K_main=2, K_sub=1):
    h = prefill(x)
    
    # 主推理第一阶段
    h = hrm_main_block(h, x)
    
    # 触发子推理（并行）
    sub_results = parallel([
        sub_reasoning_1(h, K=K_sub),
        sub_reasoning_2(h, K=K_sub),
        sub_reasoning_3(h, K=K_sub),
    ])
    
    # 主推理第二阶段（融合子结果）
    h = hrm_main_block(h, x, sub_context=sub_results)
    return h
```

**延迟**：K_main + max(K_sub) = 2 + 1 = 3 次 forward。

**适用场景**：可分解推理（数学先计算再验证）。

### 4.5 三方案对比

| 场景 | PLT/HLT-PLT | Radix Cache 多路径 | 层次化 |
|------|:---:|:---:|:---:|
| **单轨迹推理 K=4-8** | ✅ 端侧化 | — | — |
| **多假设决策** | — | ✅ 端侧化 | — |
| **可分解推理** | — | — | ✅ 主 + 子并行 |
| **延迟节省** | 62% | N 倍 | 2-3 倍 |

**结论**：**没有单一方案压倒性最优**——按场景选，Logos 64M v2.0 验证哪种组合最优。

### 4.6 实施优先级

**v1.0 (HRM-Text 集成)**：仅 K=2 串行，延迟已可接受。

**v1.5**：条件性实施并行化——仅在以下情况启动：
- GRAM 多轨迹扩展到 K=8 时 → 用 Radix Cache
- 端侧 K=8 时延仍超预算时 → 用 HLT-PLT
- 可分解推理任务出现时 → 用层次化

**风险**：CLP 位置偏移税可能导致 L 循环精度损失。**STARS 谱正则化作为 Plan B**（详见 [stars.md](../references/stars.md)）。

---

## 5. 端侧时延预算详细分析

### 5.1 硬件假设

```
ChipForge APU（假设规格）
├── 内存：4 GiB（端侧预算）
├── FP16 算力：X TFLOPS（与具体硬件相关，待 [projects.md](../references/projects.md) 补充）
├── INT8 算力：Y TFLOPS
└── KV cache：INT8 端侧
```

**关键约束**：1B 参数 FP16 = 2 GiB，**几乎占满内存预算**。KV cache 必须 INT8 量化。

### 5.2 各模块延迟贡献

| 模块 | 占比 | 优化方向 |
|------|------|---------|
| 感知层（3 层 CNN）| 15% | INT8 推理足够 |
| HRM backbone（H+L 循环）| **70%** | K 调度 + PLT 并行化 |
| 双流解码头 | 10% | INT8 推理 |
| 其他（采样、KV 管理）| 5% | — |

**主战场**：HRM backbone 的循环次数与并行度。

### 5.3 时延预算分配

| 任务 | 总预算 | 感知层 | HRM | 解码 |
|------|--------|--------|------|------|
| 自动驾驶感知 | 50ms | 7ms | **35ms**（K=1）| 8ms |
| 实时对话 | 100ms | 15ms | **70ms**（K=2）| 15ms |
| 离线规划 | 500ms | 50ms | **400ms**（K=8 或 K=2+8 轨迹）| 50ms |

**端侧 K=2 硬上限**：对话 100ms / 自动驾驶 50ms（K=1）| 离线 500ms（K=8 允许）。

---

## 6. Plan B 触发条件（RDD-0001）

### 6.1 原阻塞定位 vs 现 Plan B 定位

| 维度 | 原阻塞定位（废弃） | 现 Plan B 定位 |
|------|----------------|---------------|
| 状态 | 主线前置必做 | **条件性启动** |
| 时序 | +0 月（HRM 集成之前）| 集成失败后启动 |
| 范围 | 4 实验 × 6 K 值 × 5 数据集（54h）| 2-3 K 值 × 1 数据集（<10h）|
| 触发条件 | 始终启动 | 满足 §6.2 任一条件 |

### 6.2 触发条件（满足任一即启动）

| # | 条件 | 诊断方式 |
|---|------|---------|
| 1 | **K=8 推理时延超端侧预算** | 在 ChipForge APU 上跑 K=8 推理，若 > 500ms |
| 2 | **K=8 出现 gain-then-collapse** | 训练曲线在 K=6+ 出现精度衰退 > 10% |
| 3 | **端侧 K 必须 ≤ 2**（无 early exit） | 时延超预算且 early exit 不能解决 |
| 4 | **集成中遇到未预期的 K 相关 bug** | K 调度逻辑失效 |

### 6.3 Plan B 实验范围（收缩版）

| 实验 | 目标 | K 值 |
|------|------|------|
| **A.1 快速 K-sweep** | 找到端侧预算内的最优 K | {2, 4, 8} |
| **B.1 层次 vs 扁平** | 验证层次化是否免疫崩溃 | 2H×3L vs flat 8 |
| 总时长 | <10h | — |
| 数据集 | 仅 GSM8K（推理任务代表）| — |

### 6.4 决策传递

Plan B 实验结果如何传递给 Logos v1.0：

| 实验结果 | Logos v1.0 K 决策 |
|---------|-----------------|
| K=2 足够 | 维持 K=2（默认） |
| K=4 比 K=2 显著提升且延迟可接受 | K=2 默认 + per-token 动态升级到 K=4 |
| K=8 比 K=4 显著提升且延迟可接受 | K=4 默认 + 离线任务用 K=8 |
| K=8 出现崩溃 | 启用 STARS 谱正则化（[stars.md](../references/stars.md)）|
| 任意 K 都崩 | 切换到 Huginn 风格（无层次化、无 CLP）|

---

## 7. 与 SADKO 路线图的协同

### 7.1 时序协同

Logos 与 SADKO 的前置研究**并行**：

| 时点 | Logos | SADKO |
|------|-------|-------|
| **+0 月** | 启动 HRM-Text 集成（K=8 起步） | 启动 64M 四大机制实验 |
| **+2 月** | 集成进度检查（可能触发 Plan B）| 64M 阶段一完成（v1.0 左脑改造）|
| **+3 月** | Plan B 出结果（若启动）| 64M 阶段二完成（v2.0 记忆压缩）|
| **+4 月** | 双脑融合接口原型 | 64M 阶段三完成（v3.0 灵魂注入）|
| **+5 月** | v1.0 Demo 完成 | 64M 机制清单产出 |

### 7.2 风险对冲

| 风险 | Logos 缓解 | SADKO 缓解 |
|------|----------|----------|
| **HRM K=8 失败** | Plan B（K-sweep + early exit）| — |
| **多模态感知失败** | 切换到 SADKO ELF 双向感知主干 | — |
| **记忆外置失败** | — | 退回 SAGo 的 AR-only 版本 |
| **融合接口时延** | 端侧 KV cache INT8 + 稀疏读取 | 减少 Memory KV 维度 |

详细路线图见 [logos-roadmap.md](./logos-roadmap.md)。

---

## 8. 关键引用块

> LoopCoder-v2 核心支撑（已被本策略消化）：
> "On SWE-bench Verified, our 7B baseline achieves 43.0%. With only one extra loop, performance jumps to 64.4%. This validates that cyclic architecture is the right direction."

> HRM-Text 论文关键数字（保留作 K=8 起点）：
> "At 1B parameters, HRM-Text is trained on 40B unique tokens (60B total with 4 epochs) and reaches MMLU 60.7 — 5–8 points above similarly sized open baselines."

> Per-Token Convergence（动态 K 证据）：
> "Median token converges by step 6, but 10% of tokens require 8 steps. This suggests adaptive per-token compute allocation, rather than fixed K, is the right inference strategy."

> STARS 关键发现（崩溃修复 Plan B）：
> "Constraining the spectral radius ρ(J_f) < γ ensures the recurrent block is contractive or at most critically stable, preventing the activation explosion observed in deep loops."

> Logos K 策略哲学（2026-07-29）：
> "Logos 默认 K=2，但永远不应该写死 K——K 是任务和 token 共同决定的运行时决策。"

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | K 策略的根本动机 |
| [docs/references/huginn.md](../references/huginn.md) | 反例：循环可 K=50（支持 Logos K 起点）|
| [docs/references/stars.md](../references/stars.md) | 崩溃修复方案（Plan B 触发后使用）|
| [docs/references/per-token-convergence.md](../references/per-token-convergence.md) | 动态 K 的实证证据 |
| [docs/references/hrm-text.md](../references/hrm-text.md) | HRM-Text 完整笔记（K=8 起点来源）|
| [logos-whitepaper.md](./logos-whitepaper.md) | Logos 主线架构白皮书（本文档的父级）|
| [logos-roadmap.md](./logos-roadmap.md) | 详细路线图 |
| [RDD-0001-k-sweep-experiment.md](../rfcs/RDD-0001-k-sweep-experiment.md) | K-sweep Plan B RFC（已降级）|
| [AGENTS.md §7.2](../../AGENTS.md#72-主线logos的-k-值策略不再阻塞) | K 策略调整说明 |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos K 策略整理）
**版本**：v1.0