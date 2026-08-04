# AGENTS.md — LatentMind 开发手册

## 1. 项目概述

**项目**：LatentMind（原生多模态潜空间认知推理模型）
**定位**：ChipForge APU 的认知核，<1B 参数，物理 AI 端侧推理
**架构核心**：**四研究线**——**Logos**（主线：分层循环 + 多轨迹推理） + **SADKO**（探索分支：双向 + FM + FSQ） + **Hippo**（独立研究线：记忆 / KG / 检索） + **Thumos**（独立研究线：agent 内化为循环动力学）；**全部全新架构从零训练**

> 📌 **关键战略（2026-07-29 v1.4 + 2026-07-31 增补 Thumos）**：
> - 四研究线都是**全新架构**，**从零训练**权重
> - 不复用现有大模型预训练权重（HRM-Text / GRAM / Loop B 论文等都是**架构灵感参考**，不是集成组件）
> - **64M 是起点**（MiniMind3 64M Dense 作为基座参考），不是验证
> - 规模扩展路径：**64M → 300M → 1B → 1.5B**（每级独立训练，逐步扩大）
> - K 值是**超参数**，不是架构决策
> - 64M 验证计划包含**§3.5 DiscoLoop 探针 + §3.6 Loop B 备选 + §4.6 L-PLT + §4.6.9 L-PLT+Φ** 多个新机制

> 📦 **参考文档体系**（详见各文件）：
> - 总体架构：[`docs/architecture/architecture.md`](docs/architecture/architecture.md)
> - **四研究线索引**：[`docs/research/README.md`](docs/research/README.md)（顶级 + 四线 + 综合 + 调研）
> - **四线综合脉络**：[`docs/research/four-lines-synthesis.md`](docs/research/four-lines-synthesis.md)（V4 议题归并 + 跨线协调）
> - **循环+记忆系统文献谱系**：[`docs/research/loop-memory-survey.md`](docs/research/loop-memory-survey.md)（Loop A/B/C + 7 篇 Loop B 论文完整谱系）
> - **Logos 主线白皮书**：[`docs/research/logos/whitepaper.md`](docs/research/logos/whitepaper.md)
> - **Logos K 值策略**：[`docs/research/logos/k-strategy.md`](docs/research/logos/k-strategy.md)（含 §2.2.7 PLT + Φ 集成设计）
> - **Logos 路线图**：[`docs/research/logos/roadmap.md`](docs/research/logos/roadmap.md)
> - **Logos 64M 验证计划**：[`docs/research/logos/64m-validation-plan.md`](docs/research/logos/64m-validation-plan.md)（v1.5：§3.5 + §3.6 + §4.6 + §4.6.9）
> - **Logos / SADKO 双轨协调中枢**：[`docs/research/logos/sadko-64m-coordination.md`](docs/research/logos/sadko-64m-coordination.md)
> - **FSQ 配置冲突 SOP**：[`docs/research/logos/codebook-config-sop.md`](docs/research/logos/codebook-config-sop.md)
> - **Logos 消费 Hippo FSQ 条件设计**：[`docs/research/logos/fsq-consumption-design.md`](docs/research/logos/fsq-consumption-design.md)
> - **Logos v1.0 架构（双时间尺度对比）**：[`docs/research/logos/v1-architecture.md`](docs/research/logos/v1-architecture.md)
> - **Logos v2.0 架构（多种循环策略）**：[`docs/research/logos/v2-architecture.md`](docs/research/logos/v2-architecture.md)
> - **Logos v3.0 架构（GRAM + Radix Cache）**：[`docs/research/logos/v3-architecture.md`](docs/research/logos/v3-architecture.md)
> - **SADKO 白皮书**：[`docs/research/sadko/whitepaper.md`](docs/research/sadko/whitepaper.md)
> - **SADKO 多模态原生设计**：[`docs/research/sadko/multimodal-native.md`](docs/research/sadko/multimodal-native.md)
> - **SADKO 文档体系审查**：[`docs/research/sadko/open-issues.md`](docs/research/sadko/open-issues.md)（含 B-09 Compressive 1D Conv 备选）
> - **SADKO 64M 验证计划**：[`docs/research/sadko/64m-validation-plan.md`](docs/research/sadko/64m-validation-plan.md)
> - **Hippo 独立研究线**：[`docs/research/hippo/README.md`](docs/research/hippo/README.md)（含胼胝体接口契约 I1-I6 + INV-1-6）
> - **Hippo 记忆内容架构**：[`docs/research/hippo/memory-architecture.md`](docs/research/hippo/memory-architecture.md)
> - **Hippo KG 压缩生长**：[`docs/research/hippo/graph-growth.md`](docs/research/hippo/graph-growth.md)
> - **Hippo FM 检索提取**：[`docs/research/hippo/retrieval-extraction.md`](docs/research/hippo/retrieval-extraction.md)
> - **Thumos 独立研究线**：[`docs/research/thumos/README.md`](docs/research/thumos/README.md)（含赫尔墨斯接口契约 H1-H6 + INV-1-6）
> - **Thumos 三方向骨架**：[`docs/research/thumos/intent-internalization.md`](docs/research/thumos/intent-internalization.md) / [fork-join-internalization.md](docs/research/thumos/fork-join-internalization.md) / [offline-online-dynamics.md](docs/research/thumos/offline-online-dynamics.md)
> - **Phase 0 实施指南**：[`docs/implementation/phase-0-implementation-guide.md`](docs/implementation/phase-0-implementation-guide.md)
> - **Phase 0 失败回退 SOP**：[`docs/implementation/phase-0-recovery-sop.md`](docs/implementation/phase-0-recovery-sop.md)
> - **实施文档索引**：[`docs/implementation/README.md`](docs/implementation/README.md)
> - **外部研究笔记**：[`docs/references/README.md`](docs/references/README.md)
>   - Loop A（循环架构）：HRM-Text / GRAM / TRM / Huginn / STARS / LoopCoder-v2 / **DiscoLoop**
>   - Loop B（持久记忆）：**Memorizing Transformers / Compressive Transformers / StreamingLLM / InfLLM / AutoCompressors / RMT / Landmark Attention**
>   - 其他：RRM-A / RRM-B 谱系 / Huginn / STARS / Per-Token Convergence
> - **内部 R&D 提案**：[`docs/rfcs/README.md`](docs/rfcs/README.md)
> - **相关项目元信息**：[`docs/references/projects.md`](docs/references/projects.md)（ChipForge / HydraForge / AgenticLlama / minimind）

---

## 2. 技术架构（v1.0）

### v1.0 架构（Demo）

```
图像输入
  ↓
原生统一感知层（3层 CNN，~150M） ← 从零训练
  ↓
Logos 全新架构 backbone（~850M） ← 从零训练
  ↓
语义图生成（轻量 DiT，~50M）
```

### v1.5 架构（完整方案）

```
图像 / 文本 / 知识图谱
  ↓
原生统一感知层（3层 CNN，~150M）
  ↓
Logos 分层递归潜空间引擎（~750M）—— 从零训练
  ├── H/L 双时间尺度循环（数学同构于 SADKO Split-GQA）
  ├── Per-Token 早退（动态 K）
  └── 多轨迹推理（GRAM 思路参考）
  ↓
双流解码层（~100M）
  ├── 语言解释头（PrefixLM）
  └── 生成式语义头（轻量 DiT / Flow Matching）
```

> 📌 **新架构基础**：Logos 全程从零训练，MiniMind3 64M Dense 仅作为**基座参考**（v1.0 64M 阶段），不依赖于任何外部预训练权重。

---

## 3. 训练阶段

| 阶段 | 内容 | 目标 |
|------|------|------|
| **阶段一** | 潜空间自监督预热 | 让模型学会"机器潜空间语言" |
| **阶段二** | 仅回答目标 + 双任务对齐 | 只对最终输出计算 loss |
| **阶段三** | 多轨迹推理微调 | （GRAM 思路参考，非集成）|

---

## 4. 关键技术

### 4.1 Logos 全新架构核心
- 分层双时间尺度循环（H/L 借鉴 HRM-Text，独立设计）
- 异构 RoPE（Split-GQA 思路，数学同构于 H/L）
- Per-Token 早退（动态 K 调度）
- 多轨迹推理（GRAM 思路参考）
- 全程从零训练

### 4.2 SADKO 探索分支核心
- 双向注意力 + Flow Matching（FM）
- FSQ 离散化（替代 VQ-VAE）
- 异构双脑（左 AR + 右 FM）
- 扩散对齐桥梁（训练时 Teacher-Student，推理时 Zero-Overhead）
- 全程从零训练

### 4.3 原生统一感知
- 像素和文字在同一 latent space 联合迭代
- 3 层 CNN 轻量感知层
- 无独立视觉编码器（VE-free）

---

## 5. 开发原则

### 5.1 **从零训练原则**
- Logos 和 SADKO 都是**全新架构**，从零训练
- 不复用现有大模型的预训练权重
- MiniMind3 64M Dense 作为基座**参考**（仅 64M 阶段），不作为预训练权重

### 5.2 阶段验证
- 每个阶段有明确的验证指标
- 先验证 64M 架构机制，再扩展到 300M → 1B → 1.5B
- 最后验证多模态融合

### 5.3 量化方案
- 训练：FP16（不用 Q4）
- 端侧推理：感知层 INT8/Q4，backbone FP16，动作头 INT8
- 目标 < 1 GiB 激活

---

## 6. 里程碑

| 时间 | 里程碑 | 状态 |
|------|--------|:---:|
| +0 月 | Logos 64M 训练 + 架构验证（MiniMind3 基座参考）| ⏳ 待启动 |
| +2 月 | 64M 验证报告（机制清单 + 循环策略 + 量化可行性）| ⏳ 待开始 |
| +3 月 | Logos 300M 训练启动 | ⏳ 规划中 |
| +6 月 | Logos 1B 训练 + 端侧集成 | ⏳ 规划中 |
| +9 月 | Logos 1B 端侧 Demo | ⏳ 规划中 |
| +12 月 | Logos 1B + SADKO 1.5B 完整融合 | ⏳ 规划中 |

**详细路线图**：[docs/research/logos/roadmap.md](docs/research/logos/roadmap.md)

---

## 7. 研究路线分工（双轨 + 两条独立研究线）— 2026-07-29 战略决策 + 2026-07-31 增补

> **背景**：Logos / SADKO / Hippo / Thumos 四线都是全新架构，**从零训练**。四条路线从 64M 起点并行启动，逐步扩展。

### 7.1 四条路线的定位

| 维度 | **Logos 主线** | **SADKO 探索分支** | **Hippo 独立研究线** 🆕 | **Thumos 独立研究线** 🆕 |
|------|--------------|------------------|---------------------------|---------------------------|
| **定位** | 推理 + 决策 | 感知 + 记忆 + 知识 + 多模态 | 记忆 / KG / 检索 | agent 内化为循环动力学 |
| **backbone** | 分层双时间尺度循环（H/L 借鉴）| AR-Native + Split-GQA | 双向 + FM + FSQ | V4 三 Session 循环结构模式 |
| **核心范式** | 循环 + 多轨迹采样 | 异构双脑 + 扩散桥梁 | FSQ 256 离散码本 | Intent/Fork-Join/触发内化 |
| **数学偏置** | 序列 + 收敛 + 不确定性 | 空间 + 流形 + 离散锚点 | 流形 + 离散索引 | 状态机 + 反馈回路 |
| **目标硬件** | ChipForge APU 端侧 | 服务器训练 + 端侧 KV 缓存 | 端侧 KV cache | 端侧 forward pass |
| **哲学隐喻** | 大脑皮层（理性思考）| 海马体（流形记忆）| 知识召回（海马体提取）| 激情/驱动力（Thumos 主动执行）|
| **接口契约** | Nano-WM Gate | 胼胝体契约（I1-I6）| 自指（属 SADKO 延伸）| 赫尔墨斯契约（H1-H6）|

### 7.2 Logos 主线 K 值策略

**K 值是超参数，不是架构决策**。具体 K 值不重要——重要的是循环架构方向是否正确。

**借鉴的架构思想**：
- HRM-Text 的循环方向（已由 LoopCoder-v2 验证正确）
- GRAM 的多轨迹思路（仅作参考，不集成）
- PLT 的并行化机制（用于端侧 K 调度）
- Split-GQA 的异构 RoPE（数学同构于 H/L）

**端侧 K 策略**：
- K > 4 串行循环在端侧实际价值≈0（K=8 = 400ms 不可用）
- 通过 PLT / Radix Cache / 层次化策略让 K>4 端侧化
- 具体 K 值由 64M 验证决定，不是预先固定

### 7.3 SADKO 探索分支的定位升级

> **2026-07-29 增补**：SADKO 从"文本认知推理备选"升级为"**多模态流形记忆与感知**"核心研究方向。详细论证见 [SADKO 多模态原生设计](docs/research/sadko/multimodal-native.md)。

**升级依据**（第一性原理）：
- 双向注意力（无因果 Mask）天然处理流形
- Flow Matching（ODE 可逆变换）天然处理压缩-解压对称
- FSQ 离散化天然支持跨模态共享字典

### 7.4 关键原则

1. **从零训练原则**：两条路线均从零训练，不复用外部预训练权重
2. **64M 起点原则**：64M 是设计起点（不是验证），从 64M 开始逐步扩展
3. **机制优先**：64M 阶段产出《机制清单》比"最佳性能"更重要
4. **融合优先**：两条路线最终应融合（Hippo KV → HRM Cross-Attention），而非互斥
5. **K 自适应**：K 不是固定超参，根据任务难度 + per-token 收敛状态动态调整

### 7.5 决策时间表

| 时点 | 决策 |
|------|------|
| **+0 月** | 启动 Logos 64M 训练 + SADKO 64M 训练（并行）|
| **+2 月** | Logos 64M 验证报告 + SADKO 64M 机制清单 |
| **+3 月** | 决策：Logos 64M 是否升格为 300M 训练 |
| **+6 月** | Logos 1B 训练启动 |
| **+9 月** | Logos 1B 端侧 Demo |
| **+12 月** | Logos 1B + SADKO 1.5B 完整融合 |

---

## 8. 关键引用块（2026-07-31 增补）

> **Thumos 启动事件**（2026-07-31）：
> 第四条研究线 Thumos 正式启动，将 V4 议题中"三 Session + Fork-Join + 智能触发"作为**模型循环结构模式**实现（而非外部 agent 调度），与 HydraForge 形成"内化 + 外化协同"。详见 [docs/research/thumos/README.md](docs/research/thumos/README.md)。

> **Loop B 调研完成**（2026-07-31）：
> 项目完成 7 篇 Loop B 论文完整覆盖（Memorizing / Compressive / StreamingLLM / InfLLM / AutoCompressors / RMT / Landmark），形成端侧三档备选方案：StreamingLLM（最轻量）/ InfLLM（中等）/ Hippo FM+FSQ（完整）。详见 [docs/research/loop-memory-survey.md](docs/research/loop-memory-survey.md)。

> **DiscoLoop × PLT 跨流派发现**（2026-07-31）：
> 用户识别 PLT 后一轮输入 `shift(h^(r-1)_{i-1})` 与 DiscoLoop 的 H^(k) 面临**同一类问题**（h 偏离 E 分布），但**触发维度不同**（纵向漂移 vs 横向错位）。两者本质相同（f_θ 消费分布失配），Φ 通道是通用修复算子。已在 Logos 64M 验证计划加入 §4.6.9 L-PLT + Φ 联合实验。

> **关键架构决策**（2026-07-29）：
> - **Logos 和 SADKO 都是全新架构**，**从零训练**权重
> - **不复用** HRM-Text / GRAM / 等任何外部预训练权重
> - **64M 是设计起点**（不是验证），从 64M 开始逐步扩展
> - **K 值是超参数**，由 64M 验证决定，不是架构决策