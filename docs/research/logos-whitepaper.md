# Logos / 逻各斯 主线架构白皮书

> **一句话定位**：LatentMind 主线架构 Logos——以"分层循环 + 多轨迹采样 + 端侧推理"三件套，构建 **<1B 参数、ChipForge APU 部署的物理 AI 认知核**，专攻**推理 + 决策**任务。
> **性质**：Logos 主线架构的全要素合并稿，涵盖核心哲学、架构蓝图、双轨分工、训练策略、端侧集成、风险降级与演进路径
> **最后更新**：2026-07-29
> **命名来源**：希腊哲学 Logos（λόγος）——赫拉克利特"万物运行的理性原则"，斯多葛学派"宇宙之理"，基督教神学"太初有道"。与 SADKO（俄罗斯民间传说，海洋奇迹的探索者）形成哲学抽象 vs 民间叙事的对仗。

---

## 0. Logos 与 SADKO 的关系

LatentMind 项目采用**双轨制**：Logos（主线）+ SADKO（探索分支）。

| 维度 | **Logos 主线** | **SADKO 探索分支** |
|------|--------------|------------------|
| **定位** | 推理 + 决策 | 感知 + 记忆 + 知识 + 多模态 |
| **backbone** | HRM-Text（1.15B 层次化循环）| AR-Native + Split-GQA + ELF |
| **核心范式** | 循环 + 多轨迹采样 | 双向注意力 + Flow Matching + FSQ |
| **数学偏置** | 序列 + 收敛 + 不确定性 | 空间 + 流形 + 离散锚点 |
| **目标硬件** | ChipForge APU 端侧 | 服务器训练 + 端侧 KV 缓存 |
| **哲学隐喻** | 大脑皮层（理性思考）| 海马体（流形记忆）|

**详细分工论证**：见 [sadko-multimodal-native.md](./sadko-multimodal-native.md)。

**融合接口**：SADKO 训练后冻结的 ELF Memory KV 通过 **Cross-Attention** 喂给 Logos 主线 HRM 的中间层。

---

## 1. 核心设计哲学（Core Philosophy）

Logos 确立三大底层哲学基石：

### 1.1 循环即推理（Loop is Reasoning）

**命题**：物理 AI 的认知推理，本质上是在潜空间内**多次循环迭代同一函数**，逐步逼近"理性答案"。

**技术映射**：
- HRM-Text 的双时间尺度循环（H 慢 + L 快，2H×3L=8 步）= 物理参数 1B，但**等效计算深度** K 倍
- 每次循环 $z^{(k+1)} = f(z^{(k)}, x)$ 都对前一状态做"反思-精炼"
- 收敛到不动点 = 推理完成

**第一性原理支撑**：
- 数学不动点定理（Banach）：压缩映射必收敛到唯一不动点
- 哲学 Logos（赫拉克利特）：理性（Logos）是万物运行的恒定原则——而 HRM 的循环正是寻找"理性答案"的数学过程

### 1.2 多轨迹即决策（Multi-Trajectory is Decision）

**命题**：在多假设场景中（路口左转/右转、规划路径 A/B/C），单次确定性推理不够，**必须采样多条潜空间轨迹并综合**。

**技术映射**：
- GRAM 的随机潜轨迹（Stochastic Latent Trajectories）= 在 K 步循环中注入 ε ~ N(μ, σ²I)
- 每条轨迹是一个潜在决策，综合 K 条轨迹的概率分布得到最终决策
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
| **循环深度 K** | **≤ 2**（默认）| K=2 单 token ≤ 100ms；K=8 超 400ms 不可接受 |
| **首 token 延迟** | < 200ms | 自动驾驶场景硬约束 |
| **KV 缓存** | 端侧 INT8 | 减少带宽 |

**关键论证**：详见 [logos-k-strategy.md](./logos-k-strategy.md)。

---

## 2. 架构蓝图与核心组件（Architecture Blueprint）

Logos v1.5 完整架构：

```
图像 / 文本 / 知识图谱
        ↓
原生统一感知层（3 层 CNN，~150M）
        ↓
分层递归潜空间引擎（~750M）—— HRM-Text
        ├── H 慢速（全局逻辑、长程因果）
        ├── L 快速（局部特征、细粒度对齐）
        └── 多轨迹扰动（GRAM 注入到 H 模块）
        ↓
双流解码层（~100M）
        ├── 语言解释头（PrefixLM）
        └── 生成式语义头（轻量 DiT / Flow Matching）
        ↓
动作 / 语义图 / 自然语言解释
```

### 2.1 模块 A：原生统一感知层（~150M）

**设计**：借鉴商汤 SenseNova U1 的 NEO-Unify 思想，极轻量 3 层 CNN 将图像 Patch 和文本 Token 映射到同一潜空间。

**关键特性**：
- 无独立视觉编码器（VE-free）
- 像素和文字从一开始就在同一空间"对话"
- 消除跨模态翻译的信息损耗

**与 SADKO 的差异**：Logos 的感知层是**单向前馈**（简单 3 层 CNN），不做循环；SADKO 的 ELF 是**双向 + FM** 流形压缩——这是"快速感知 vs 流形记忆"的分工。

### 2.2 模块 B：分层递归潜空间引擎（~750M）

**设计**：HRM-Text（Hierarchical Reasoning Model）+ GRAM 多轨迹扰动。

**HRM 双时间尺度结构**（详见 [hrm-text.md §3.2](../references/hrm-text.md#32-hrm-循环结构)）：

```python
# HF Transformers 模型类原文
z_H = embed(input_ids) * embedding_scale
z_L = z_L_init.expand_as(z_H)
for _ in range(H_cycles):     # 2
    for _ in range(L_cycles): # 3
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)
return z_H
```

**双时间尺度作用**：
- **H 慢速**（2 cycles）：全局逻辑、长程因果、"生成式语义"的宏观布局
- **L 快速**（每 H 周期 3 cycles）：局部高频特征、边缘、小目标和细粒度对齐

**机器潜空间语言**：输入信号进入后，H/L 循环 K 次（如默认 8 次），模型在潜空间内"默默思考"。

**稳定性保障**：
- **MagicNorm**（前向 PostNorm + 反向 PreNorm）——深层梯度稳定性
- **Warmup BPTT**（K=2 → K=5）—— 预热深度信用分配
- **PLT 并行化移植**（可选，详见 [logos-k-strategy.md §4](./logos-k-strategy.md)）—— K=8 推理延迟压平到单次推理延迟

### 2.3 模块 B+：GRAM 多轨迹扰动

**设计**：GRAM 变分随机化扩展 HRM 风格循环（详见 [gram.md](../references/gram.md)）。

**集成点**：在 H 模块 residual 后加可学习 μ_θ, σ_θ；注入 ε ~ N(μ, σ²I)；变分 ELBO 训练。

**作用机制**：
- **训练时**：强制 H 模块适应"噪声输入"，提升对多假设的覆盖
- **推理时**：采样 K 条潜轨迹，每条得到一个潜在决策；综合 K 条概率分布输出最安全决策

**测试时 scaling**：
- 简单任务：K=1（不注入噪声）
- 中等任务：K=2 + ε ~ N(0, σ²I)，σ 较小
- 复杂决策：K=4-8 + ε 较大，多轨迹综合

### 2.4 模块 C：双流解码层（~100M）

**设计**：潜空间推理完成后，最终 Meta-representation（元表征）送入两个轻量解码头。

| 解码头 | 输出 | 实现 |
|-------|------|------|
| **语言解释头** | 自然语言解释 / 决策说明 | PrefixLM（指令端双向 + 响应端因果）|
| **生成式语义头** | 语义图 / BEV / 控制轨迹 | 轻量 DiT 或 Flow Matching（条件生成）|

**关键设计**：
- PrefixLM 在 HRM-Text 上已验证（+5.4 MMLU，+1.3 MATH，详见 [hrm-text.md §3.1](../references/hrm-text.md#31-三大设计论文-table-3-消融已证实)）
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

详细论证见 [sadko-multimodal-native.md §1](./sadko-multimodal-native.md#1-第一性原理连续-vs-离散的数学偏置)。

### 3.2 融合接口

```
┌───────────────── Logos 主线（HRM-Text + GRAM）───────────────┐
│  推理 + 决策（循环/多轨迹的天然优势域）                         │
│  接收：感知主干输出 + SADKO ELF Memory KV (via Cross-Attention)│
│  输出：动作 / 决策 / 结构化解释                                │
└─────────────────────────┬────────────────────────────────────┘
                          │ Cross-Attention 融合
                          ↓
┌───────────────── 共享感知主干（M0）───────────────────────────┐
│  原生统一感知：3 层 CNN + 模态扩展                              │
│  由 SADKO 训练后冻结，Logos 只读取                              │
└─────────────────────────┬────────────────────────────────────┘
                          │ 共享潜空间接口
                          ↓
┌───────────────── SADKO 探索分支 ─────────────────────────────┐
│  感知 + 记忆 + 知识 + 多模态（双向/FM/FSQ 的天然优势域）        │
│  输出：共享感知主干 + ELF Memory KV Pool                       │
└─────────────────────────────────────────────────────────────┘
```

**关键比喻**："**HRM 是大脑皮层，ELF 是海马体**"——Logos 负责"思考"，SADKO 负责"记忆与感知"。

### 3.3 何时融合

**当前状态（2026-07-29）**：两条路线**独立并行**，尚未融合。

**融合启动条件**：
1. Logos 1B 端侧 Demo 跑通（+4-5 月）
2. SADKO 64M 四大机制实验产出《机制清单》（+2-3 月）
3. 双轨分工 §7.5 决策时间表确认（+4 月）

**详细协同**见 [logos-roadmap.md §4](./logos-roadmap.md#4-与-sadko-协同)。

---

## 4. 训练策略（Training Strategy）

Logos 采用**三阶段渐进训练**：

### 阶段一：潜空间自监督预热

- **目标**：让模型学会"机器潜空间语言"
- **方法**：输入多模态数据，让模型在内部循环 K 次，然后要求它重建被掩码的图像 Patch 和文本 Token
- **关键**：迫使潜空间引擎学会在内部存储和推演信息
- **基座**：复用 HRM-Text 40B tokens 预训练权重（详见 [hrm-text.md §5](../references/hrm-text.md#5-训练配方)）

### 阶段二：仅回答目标 + 双任务对齐

- **目标**：让潜空间推理的结果能够准确映射到自然语言和语义图
- **方法**：只对最终输出的自然语言和生成的语义图计算损失，不对内部循环的中间潜空间状态施加额外约束
- **关键**：让模型自由探索最优的内部潜空间计算路径

### 阶段三：多轨迹概率推理（GRAM 注入）

- **目标**：解决多模态不确定性（如路口可能左转也可能右转）
- **方法**：在 H 模块 residual 后注入 ε ~ N(μ, σ²I)，变分 ELBO 训练
- **关键**：综合多条轨迹的概率分布，生成最安全的语义图和解释
- **资源**：GRAM 论文只验证到 10-11M，**1B 验证必需**（详见 [gram.md](../references/gram.md)）

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
    ├── 分层递归潜空间引擎（HRM-Text，FP16 推理）
    └── 双流解码器（INT8 推理）
```

### 5.2 量化方案

| 组件 | 精度 | 理由 |
|------|------|------|
| **HRM-Text backbone** | FP16（**不用 Q4**）| Q4 会破坏 MagicNorm 稳定性 |
| **感知层（3 层 CNN）** | INT8 / Q4 | 视觉容忍量化损失 |
| **双流解码头** | INT8 | 输出层容忍量化 |
| **KV Cache** | INT8（端侧）| 端侧内存约束 |

### 5.3 端侧延迟预算

| 任务 | 目标延迟 | K=2 单 token | K=8 单 token |
|------|---------|-------------|--------------|
| 自动驾驶感知 | <50ms | ✅ 50ms | ❌ 200ms |
| 实时对话 | <100ms | ✅ 100ms | ❌ 400ms |
| 离线规划 | <500ms | ✅ | ✅ |

**关键约束**：**K ≤ 2 是硬预算**。详细 K 策略见 [logos-k-strategy.md](./logos-k-strategy.md)。

---

## 6. 风险与降级预案（Risk & Plan B）

### 6.1 风险矩阵

| 潜在风险 | 概率 | 影响 | 降级预案（Plan B）|
|---------|:---:|:---:|---------|
| **K=8 端侧时延超预算** | 🔴 高 | 高 | 启用 K=2 + per-token early exit（详见 [logos-k-strategy.md §3](./logos-k-strategy.md#3-k2-默认策略)）|
| **K=8 出现 gain-then-collapse** | 🟠 中 | 高 | 启用 STARS Jacobian 谱半径正则化（详见 [stars.md](../references/stars.md)）|
| **GRAM 1B 扩展失败** | 🟠 中 | 高 | 缩小到 K=4 轨迹 + 简化变分目标 |
| **感知层多模态融合失败** | 🟠 中 | 中 | 切换到 SADKO ELF 双向感知主干（融合接口）|
| **感知层 3 层 CNN 容量不足** | 🟡 低 | 中 | 扩展到 5 层 CNN，或借用 SADKO 的双向层 |
| **端侧 FP16 推理精度损失** | 🟡 低 | 中 | 局部 INT8 + 关键层 FP16 混合精度 |
| **MagicNorm 在 APU 上不稳定** | 🟡 低 | 高 | 替换为 Pre-Norm + RMSNorm 传统方案 |

### 6.2 Plan B 触发决策树

```
K=8 集成时延 > 200ms（单 token）
    ├─ YES → 启用 K=2 + per-token early exit（详细 §3）
    └─ NO → 维持 K=8
            │
            ↓
K=8 训练出现 gain-then-collapse
    ├─ YES → 启用 STARS 谱半径正则化
    └─ NO → 维持 K=8 + MagicNorm
            │
            ↓
GRAM 1B 变分训练不稳定
    ├─ YES → 退回 K=2 + ε 简化版（不训练 μ, σ）
    └─ NO → 维持 GRAM 多轨迹
```

详细触发条件见 [logos-k-strategy.md §6 Plan B 触发条件](./logos-k-strategy.md#6-plan-b-触发条件)。

---

## 7. 演进路径（Evolution Path）

Logos 从 64M 验证到 1.5B 完整方案的演进：

### 7.1 阶段零：64M 快速消融（MiniMind3，可选）

- **目标**：验证 HRM 风格循环在标准 Transformer 上的 K-精度曲线
- **范围**：仅在 K=8 集成失败时启动（Plan B，详见 [RDD-0001](../rfcs/RDD-0001-k-sweep-experiment.md)）
- **不阻塞主线**

### 7.2 阶段一：1B 端侧 Demo（v1.0，+1 至 +5 月）

- **目标**：物理 AI 端侧推理 demo
- **关键组件**：
  - HRM-Text 1B backbone（直接用 HF 权重）
  - 3 层 CNN 感知层（冻结 backbone 微调感知层）
  - 双流解码（PrefixLM + 轻量 DiT）
- **不集成 GRAM**（v1.0 不做多轨迹）
- **端侧验证**：ChipForge APU INT8 部署

### 7.3 阶段二：1B 完整推理（v1.5，+7 至 +10 月）

- **目标**：物理 AI 端侧推理 + 决策完整方案
- **新增组件**：
  - GRAM 多轨迹扰动（变分注入）
  - PLT 并行化移植（K=8 推理延迟压平）
  - per-token early exit（动态 K 调度）
- **SADKO 融合接口**（如 SADKO 64M 验证通过）：ELF Memory KV → HRM Cross-Attention

### 7.4 阶段三：1.5B 完整多模态（v2.0，+12 月）

- **目标**：物理 AI + 多模态记忆核（双核）
- **新增**：
  - Logos 1B（推理核，FP16）
  - SADKO 1.5B（多模态记忆核，INT8 KV 缓存）
  - 完整 Cross-Attention 融合接口
- **硬件**：ChipForge APU + 服务器训练

**详细路线图**：[logos-roadmap.md](./logos-roadmap.md)

---

## 8. 关键引用块（Key Quotes）

> HRM-Text 原文：
> "We present HRM-Text, a 1B-parameter hierarchical recurrent model trained from scratch using only 40B unique tokens and a budget of $1,500, achieving 60.7% MMLU, 81.9% ARC-C, 82.2% DROP, 84.5% GSM8K, and 56.2% MATH."

> GRAM 原文：
> "Stochastic latent trajectories for multi-hypothesis reasoning, with amortized variational inference."

> LoopCoder-v2 原文（关键支撑）：
> "On SWE-bench Verified, our 7B baseline achieves 43.0%. With only one extra loop, performance jumps to 64.4% (+50%). This validates that cyclic architecture is the right direction, with diminishing returns at higher K."

> Logos 哲学宣言（2026-07-29）：
> "Logos = 循环即推理 + 多轨迹即决策 + 端侧即哲学。三件套构建物理 AI 的认知核——理性的数学化身，运行在每一台边缘设备上。"

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [docs/architecture.md](../architecture.md) | 高层项目概览（已存在），Logos 是其详细技术展开 |
| [docs/references/hrm-text.md](../references/hrm-text.md) | HRM-Text 外部论文笔记（详见 §2.2）|
| [docs/references/gram.md](../references/gram.md) | GRAM 外部论文笔记（详见 §2.3）|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | K 值策略的关键论文（详见 [logos-k-strategy.md](./logos-k-strategy.md)）|
| [docs/references/huginn.md](../references/huginn.md) | 反例：循环可 K=50（支持 Logos K=2-8 假设）|
| [logos-k-strategy.md](./logos-k-strategy.md) | K 值策略与端侧可行性 |
| [logos-roadmap.md](./logos-roadmap.md) | 详细路线图与决策时间表 |
| [sadko-whitepaper.md](./sadko-whitepaper.md) | SADKO 探索分支白皮书（互补关系）|
| [sadko-multimodal-native.md](./sadko-multimodal-native.md) | 双轨分工的第一性原理论证 |
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 研究路线分工战略 |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos 主线架构整理）
**版本**：v1.0