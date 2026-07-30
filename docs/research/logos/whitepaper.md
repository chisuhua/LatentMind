# Logos / 逻各斯 主线架构白皮书

> **一句话定位**：LatentMind 主线架构 Logos——以"分层循环 + 多轨迹推理 + 端侧部署"三件套，构建 **<1B 参数、ChipForge APU 部署的物理 AI 认知核**，专攻**推理 + 决策**任务。**全新架构，从零训练**。
> **性质**：Logos 主线架构的全要素合并稿，涵盖核心哲学、架构蓝图、双轨分工、训练策略、端侧集成、风险降级与演进路径
> **最后更新**：2026-07-29（v1.4 重大更新：全新架构定位 + 移除 HRM-Text 集成假设）
> **命名来源**：希腊哲学 Logos（λόγος）——赫拉克利特"万物运行的理性原则"，斯多葛学派"宇宙之理"，基督教神学"太初有道"。与 SADKO（俄罗斯民间传说，海洋奇迹的探索者）形成哲学抽象 vs 民间叙事的对仗。

---

## 0. Logos 与 SADKO 的关系

LatentMind 项目采用**双轨制**：Logos（主线）+ SADKO（探索分支）。**两条路线都是全新架构，从零训练**。

| 维度 | **Logos 主线** | **SADKO 探索分支** |
|------|--------------|------------------|
| **定位** | 推理 + 决策 | 感知 + 记忆 + 知识 + 多模态 |
| **backbone** | 分层双时间尺度循环（H/L 借鉴，独立设计） | AR-Native + Split-GQA + ELF（独立设计） |
| **核心范式** | 循环 + 多轨迹推理 | 双向注意力 + Flow Matching + FSQ |
| **数学偏置** | 序列 + 收敛 + 不确定性 | 空间 + 流形 + 离散锚点 |
| **训练起点** | MiniMind3 64M Dense（作为基座参考） | MiniMind3 64M Dense（作为基座参考） |
| **是否复用预训练权重** | **否**（从零训练） | **否**（从零训练） |
| **架构灵感参考** | HRM-Text / GRAM / PLT / Split-GQA | Flow Matching / VQ-VAE / ELMo |
| **目标硬件** | ChipForge APU 端侧 | 服务器训练 + 端侧 KV 缓存 |
| **哲学隐喻** | 大脑皮层（理性思考）| 海马体（流形记忆）|

**详细分工论证**：见 [sadko-multimodal-native.md](../sadko/multimodal-native.md)。

**融合接口**：SADKO 训练后冻结的 ELF Memory KV 通过 **Cross-Attention** 喂给 Logos 主线 H 模块的中间层（"HRM 是大脑皮层，ELF 是海马体"）。

---

## 1. 核心设计哲学（Core Philosophy）

Logos 确立三大底层哲学基石：

### 1.1 循环即推理（Loop is Reasoning）

**命题**：物理 AI 的认知推理，本质上是在潜空间内**多次循环迭代同一函数**，逐步逼近"理性答案"。

**技术映射**：
- 借鉴 HRM-Text 的双时间尺度循环思想（H 慢 + L 快），但**独立设计**
- 每次循环 $z^{(k+1)} = f(z^{(k)}, x)$ 都对前一状态做"反思-精炼"
- 收敛到不动点 = 推理完成
- K 是超参数（不是架构决策），由 64M 验证决定

**第一性原理支撑**：
- 数学不动点定理（Banach）：压缩映射必收敛到唯一不动点
- LoopCoder-v2 验证（K=2 > K=1 +50%）证明循环架构方向正确
- 哲学 Logos（赫拉克利特）：理性（Logos）是万物运行的恒定原则——循环正是寻找"理性答案"的数学过程

### 1.2 多轨迹即决策（Multi-Trajectory is Decision）

**命题**：在多假设场景中（路口左转/右转、规划路径 A/B/C），单次确定性推理不够，**必须采样多条潜空间轨迹并综合**。

**技术映射**：
- 借鉴 GRAM 的随机潜轨迹思想（Stochastic Latent Trajectories）——**仅作架构参考，不集成其训练流程**
- 我们的实现：Per-Token 早退 + 多路径并行（Radix Cache 风格）
- 测试时 scaling：递归深度 + 并行轨迹数 = 推理计算预算

**第一性原理支撑**：
- 决策的本质是"在不确定下最大化期望收益"——概率采样是数学最优
- 哲学 Logos（斯多葛）：理性决策 = 顺应 Logos 的秩序——多轨迹采样是寻找"秩序"的工程化

### 1.3 端侧推理哲学（Edge Inference Philosophy）

**命题**：物理 AI 必须**本地实时推理**——不能依赖云端、不能高延迟、不能无限算力。

**技术映射**：

| 约束 | 值 | 推导 |
|------|------|------|
| **参数总量** | <1B | ChipForge APU 内存预算 |
| **激活参数** | <1B | FP16 推理显存 < 2 GiB |
| **循环深度 K** | **超参数**（由 64M 验证决定，不是架构选择）| K 是具体数值不重要 |
| **首 token 延迟** | < 200ms | 自动驾驶场景硬约束 |
| **KV 缓存** | 端侧 INT8 | 减少带宽 |
| **并行化策略** | PLT / Radix Cache / 层次化 | 突破 K 端侧限制 |

**关键论证**：详见 [k-strategy.md](./k-strategy.md)。

---

## 2. 架构蓝图与核心组件（Architecture Blueprint）

Logos v1.5 完整架构（**全新架构，从零训练**）：

```
图像 / 文本 / 知识图谱
        ↓
原生统一感知层（3 层 CNN，~150M） ← 从零训练
        ↓
Logos 分层递归潜空间引擎（~750M） ← 从零训练
        ├── H/L 双时间尺度循环（数学同构于 Split-GQA）
        ├── Per-Token 早退（动态 K 调度）
        └── 多轨迹并行推理（Radix Cache 风格）
        ↓
双流解码层（~100M） ← 从零训练
        ├── 语言解释头（PrefixLM）
        └── 生成式语义头（轻量 DiT / Flow Matching）
        ↓
动作 / 语义图 / 自然语言解释
```

### 2.1 模块 A：原生统一感知层（~150M）

**设计**：借鉴商汤 SenseNova U1 的 NEO-Unify 思想，极轻量 3 层 CNN 将图像 Patch 和文本 Token 映射到同一潜空间表示中。**从零训练**。

**特点**：
- 无独立视觉编码器（VE-free）
- 像素和文字从一开始就在同一空间"对话"
- 消除了跨模态翻译的信息损耗

**与 SADKO 的差异**：Logos 的感知层是**单向前馈**（简单 3 层 CNN），不做循环；SADKO 的 ELF 是**双向 + FM** 流形压缩——这是"快速感知 vs 流形记忆"的分工。

### 2.2 模块 B：Logos 分层递归潜空间引擎（~750M）

**设计**：Logos 全新架构，**从零训练**。借鉴 HRM-Text 的双时间尺度循环思想 + SADKO Split-GQA 的异构 RoPE 思路。

**H/L 双时间尺度结构**：

```python
# Logos H/L 双时间尺度循环（K 是超参数）
z_H = embed(input_ids) * embedding_scale
z_L = z_L_init.expand_as(z_H)
for h in range(H_cycles):     # 默认 2
    for l in range(L_cycles): # 默认 3
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)
return z_H
```

**双时间尺度的数学同构**：

| 维度 | SADKO Split-GQA | Logos H/L | 数学本质 |
|------|----------------|----------|---------|
| 长程/慢速 | Static KV heads (base=500k) | H module | 长程信息保留 |
| 短程/快速 | Dynamic KV heads (base=10k) | L module | 短程信息处理 |
| 双时间尺度 | head 维度分裂 | block 维度分裂 | 同一洞察 |

**关键整合**（v1.0 64M 验证）：H/L block 内部采用 Split-GQA 风格——H block 用更多 Static heads（长程），L block 用更多 Dynamic heads（短程）。**双时间尺度在 block 维度和 head 维度同时存在**。

**稳定性保障**：
- MagicNorm 思想（前向 PostNorm + 反向 PreNorm）
- Per-Token 早退（动态 K 调度）
- 异构 RoPE（base=10k vs 500k）

### 2.3 模块 B+：多轨迹并行推理（端侧化关键）

**设计**：借鉴 GRAM 随机潜轨迹思想，**但完全独立实现**——不用 GRAM 的训练流程。

**核心机制**：
- **Per-Token 早退**：借鉴 [Per-Token Convergence 论文](../references/per-token-convergence.md)（90% token 6 步收敛）
- **Radix Cache 多路径并行**：借鉴 [PLT 架构](../references/loopcoder-v2.md) 的并行循环思想，扩展为多路径版本
- **层次化推理**：主 + 子并行（适合可分解推理任务）

**与原 GRAM 思想的差异**：
- ❌ 不集成 GRAM 的变分 ELBO 训练流程
- ❌ 不使用 GRAM 的 μ, σ 可学习参数
- ✅ 借鉴"多轨迹综合决策"的**思想**，独立实现端侧友好版本
- ✅ 用 Per-Token 早退 + 多路径并行实现"测试时 scaling"

### 2.4 模块 C：双流解码层（~100M）

**设计**：潜空间推理完成后，最终 Meta-representation（元表征）送入两个轻量解码头。**从零训练**。

| 解码头 | 输出 | 实现 |
|-------|------|------|
| **语言解释头** | 自然语言解释 / 决策说明 | PrefixLM（指令端双向 + 响应端因果）|
| **生成式语义头** | 语义图 / BEV / 控制轨迹 | 轻量 DiT 或 Flow Matching（条件生成）|

**关键设计**：
- PrefixLM 借鉴 HRM-Text（+5.4 MMLU，+1.3 MATH），但**不依赖 HRM-Text 权重**
- 双流解码使 Logos 能同时输出"语言解释"和"动作/语义图"——物理 AI 必需

---

## 3. 与 SADKO 的双轨分工（Dual-Track Division）

Logos 与 SADKO 的关系：**互补 + 融合**，非竞争。

### 3.1 第一性原理分工

| 任务类型 | 数学偏置 | 适合范式 | 归属 |
|---------|---------|---------|------|
| 结构化推理 (math/code) | 离散、序列、收敛 | 循环 | **Logos** |
| 决策规划 (多假设) | 不确定性、概率分布 | 多轨迹 | **Logos** |
| 视觉感知 | 空间连续、流形 | 双向 + FM | **SADKO** |
| 跨模态对齐 | 异构空间投影 | FSQ 锚点 | **SADKO** |
| 长期记忆 | 压缩、解压对称 | 双向 + FM | **SADKO** |
| 知识库 | 离散索引、检索 | FSQ 离散 | **SADKO** |

详细论证见 [sadko-multimodal-native.md §1](../sadko/multimodal-native.md#1-第一性原理连续-vs-离散的数学偏置)。

### 3.2 融合接口（远期）

```
┌───────────────── Logos 主线（H/L 分层循环）───────────────┐
│  推理 + 决策（H/L 双时间尺度的天然优势域）                       │
│  接收：感知主干输出 + SADKO ELF Memory KV (via Cross-Attention) │
│  输出：动作 / 决策 / 结构化解释                                  │
└─────────────────────────┬────────────────────────────────────┘
                          │ Cross-Attention 融合
                          ↓
┌───────────────── 共享感知主干（M0）───────────────────────────┐
│  原生统一感知：3 层 CNN + 模态扩展                                  │
│  由 SADKO 训练后冻结，Logos 只读取                                  │
└─────────────────────────┬────────────────────────────────────┘
                          │ 共享潜空间接口
                          ↓
┌───────────────── SADKO 探索分支 ─────────────────────────────┐
│  感知 + 记忆 + 知识 + 多模态（双向/FM/FSQ 的天然优势域）             │
│  输出：共享感知主干 + ELF Memory KV Pool                            │
└─────────────────────────────────────────────────────────────┘
```

**关键比喻**："**HRM 是大脑皮层，ELF 是海马体**"——Logos 负责"思考"，SADKO 负责"记忆与感知"。

### 3.3 何时融合

**当前状态（2026-07-29）**：两条路线**独立并行**，尚未融合。

**融合启动条件**：
1. Logos 1B 训练完成（+6 月）
2. SADKO 64M 四大机制实验产出《机制清单》（+2-3 月）
3. 双轨分工 §7.5 决策时间表确认（+3 月）

---

## 4. 训练策略（Training Strategy）

Logos 采用**三阶段渐进训练**——**完全从零训练**，不依赖任何预训练权重。

### 阶段一：潜空间自监督预热（64M）

- **目标**：让新架构学会"机器潜空间语言"
- **方法**：基于 MiniMind3 64M Dense 基座，改造为 Logos H/L 架构，**完全重新训练**
- **关键**：迫使新潜空间引擎学会在内部存储和推演信息
- **数据**：4B tokens（与 MiniMind3 原始 Pretrain 数据对齐，作为起点）

### 阶段二：仅回答目标 + 双任务对齐

- **目标**：让新架构的潜空间推理能映射到自然语言和语义图
- **方法**：只对最终输出计算损失
- **关键**：让模型自由探索最优的内部潜空间计算路径

### 阶段三：多任务对齐（规模扩展）

- **目标**：让多模态输出与决策任务对齐
- **方法**：在多任务数据集上微调
- **关键**：保留 Logos 推理的涌现能力

---

## 5. 与 ChipForge APU 的集成（Edge Integration）

### 5.1 硬件约束

```
ChipForge APU
├── RISC-V 控制核
├── CUDA SIMT 计算核
├── Tensor Core 矩阵加速
└── Logos 认知核（软件栈最顶层）
    ├── 原生感知层（CNN，INT8 推理）
    ├── 分层递归潜空间引擎（Logos，FP16 推理）
    └── 双流解码器（INT8 推理）
```

### 5.2 量化方案

| 组件 | 精度 | 理由 |
|------|------|------|
| **Logos backbone** | FP16（**不用 Q4**）| Q4 会破坏 MagicNorm 稳定性 |
| **感知层（3 层 CNN）** | INT8 / Q4 | 视觉容忍量化损失 |
| **双流解码头** | INT8 | 输出层容忍量化 |
| **KV Cache** | INT8（端侧）| 端侧内存约束 |

### 5.3 端侧延迟约束

| 任务 | 目标延迟 | 单 token K=2 | 单 token K=8（理论上）|
|------|---------|-------------|------------------|
| 自动驾驶感知 | <50ms | ✅ 50ms | ❌ 400ms |
| 实时对话 | <100ms | ✅ 100ms | ❌ 400ms |
| 离线规划 | <500ms | ✅ | ✅ |

**关键约束**：K>4 串行循环在端侧实际价值≈0（K=8 不可用）。需要用 PLT / Radix Cache / 层次化等并行化策略突破 K 限制（详见 [k-strategy.md](./k-strategy.md)）。**具体 K 值由 64M 验证决定，不是预先固定**。

---

## 6. 风险与降级预案（Risk & Plan B）

### 6.1 风险矩阵

| 潜在风险 | 概率 | 影响 | 降级预案（Plan B）|
|---------|:---:|:---:|---------|
| **64M 训练崩溃** | 🔴 高 | 高 | 启用 MagicNorm 思想 + 调整初始化 |
| **新架构 K 值过大** | 🟠 中 | 高 | 用 PLT 并行化降到 1x 延迟（详见 §6.2）|
| **多轨迹决策质量差** | 🟠 中 | 中 | 退回 Per-Token 早退单一路径 |
| **感知层多模态融合失败** | 🟠 中 | 中 | 切换到 SADKO ELF 双向感知主干（融合接口）|
| **端侧 FP16 推理精度损失** | 🟡 低 | 中 | 局部 INT8 + 关键层 FP16 混合精度 |
| **64M 收敛失败** | 🟡 低 | 高 | 退回标准 Transformer 基线验证 MiniMind3 训练流程 |

### 6.2 Plan B 触发决策树

```
Logos 64M 训练收敛？
    ├─ YES → 进入 64M 验证 + 出《机制清单》
    └─ NO  → 启用人 MagicNorm 思想 + 调整初始化
            ├─ 收敛 → 继续
            └─ 不收敛 → 退回标准 Transformer

K>4 在端侧不可用？
    ├─ YES → 启用 PLT 并行化（HLT-PLT）
    └─ NO  → 维持串行 K 循环

多轨迹决策质量差？
    ├─ YES → 退回 Per-Token 早退单一路径
    └─ NO  → 维持多轨迹并行
```

---

## 7. 演进路径（Evolution Path）

Logos 从 64M 起点到 1.5B 完整方案的演进（**每级独立训练，从零开始**）：

### 7.1 阶段一：64M 起点（+0 至 +2 月）

**目标**：在 MiniMind3 64M Dense 基座上，**完全重新训练**为 Logos H/L 架构，验证核心机制。

**关键产出**：
- 《64M 机制清单》——比"最佳性能"更重要
- 决定 300M 的具体架构细节
- 推荐 K 值范围（不是固定值）

**详细计划**：[64m-validation-plan.md](./64m-validation-plan.md)

### 7.2 阶段二：300M 规模验证（+3 至 +4 月）

**目标**：从 64M 扩展到 300M，重新训练（不依赖 64M 权重作为预训练，仅借鉴结构）。

**新增能力**：
- 更多循环次数（K 适当增加）
- 更多训练 tokens（~30B）
- 端侧量化验证（INT8）
- 双流解码层完整训练

### 7.3 阶段三：1B 端侧推理核（+6 至 +9 月）

**目标**：物理 AI 端侧推理芯片上的 Logos 1B。**从零训练**，不依赖 300M 权重作为预训练（仅借鉴架构与超参）。

**关键能力**：
- ChipForge APU 端侧部署
- INT8 量化（感知层 + 解码头）
- 与 SADKO 双流融合（决策 + 记忆）

### 7.4 阶段四：1.5B 完整融合（+12 月）

**目标**：Logos 1B（推理核）+ SADKO 1.5B（多模态记忆核）完整融合。

**硬件**：
- Logos 1B：ChipForge APU 端侧（FP16）
- SADKO 1.5B：服务器训练 + 端侧 KV 缓存（INT8）
- Cross-Attention 融合接口

**详细路线图**：[roadmap.md](./roadmap.md)

---

## 8. 关键引用块（Key Quotes）

> **Logos 核心战略（2026-07-29 v1.4）**：
> "Logos 是全新架构，从零训练。HRM-Text / GRAM 是**架构灵感参考**，不集成其权重。64M 是起点，逐步扩展到 300M / 1B / 1.5B。K 值是超参数，由 64M 验证决定，不是架构选择。"

> LoopCoder-v2 关键支撑：
> "On SWE-bench Verified, our 7B baseline achieves 43.0%. With only one extra loop, performance jumps to 64.4%. This validates that cyclic architecture is the right direction."

> 第一性原理双轨分工：
> "循环/多轨迹偏置'序列+收敛+不确定性' → 推理+决策（Logos）；双向/FM/FSQ 偏置'空间+流形+离散锚点' → 感知+记忆+知识+多模态（SADKO）。"

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [docs/architecture.md](../architecture.md) | 高层项目概览（已存在），Logos 是其详细技术展开 |
| [k-strategy.md](./k-strategy.md) | K 值策略与端侧可行性 |
| [roadmap.md](./roadmap.md) | 详细路线图与决策时间表 |
| [64m-validation-plan.md](./64m-validation-plan.md) | 64M 起点验证计划 |
| [v1-architecture.md](./v1-architecture.md) | v1.0 双时间尺度对比施工图 |
| [v2-architecture.md](./v2-architecture.md) | v2.0 多种循环策略详细 |
| [v3-architecture.md](./v3-architecture.md) | v3.0 多轨迹并行详细 |
| [docs/references/hrm-text.md](../references/hrm-text.md) | HRM-Text 论文笔记（仅作架构参考，不集成权重）|
| [docs/references/gram.md](../references/gram.md) | GRAM 论文笔记（仅作架构参考，不集成）|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | K 值策略关键论文 |
| [docs/references/huginn.md](../references/huginn.md) | 反例：循环可 K=50 |
| [docs/references/stars.md](../references/stars.md) | 崩溃修复方案 |
| [docs/research/sadko/whitepaper.md](../sadko/whitepaper.md) | SADKO 探索分支白皮书（互补关系）|
| [docs/research/sadko/multimodal-native.md](../sadko/multimodal-native.md) | 双轨分工的第一性原理论证 |
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 研究路线分工战略 |

---

**最后更新**：2026-07-29（v1.4 重大更新）
**作者**：来自工作流（Logos 主线架构整理）
**版本**：v1.4