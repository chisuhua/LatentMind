# AGENTS.md — LatentMind 开发手册

## 1. 项目概述

**项目**：LatentMind（原生多模态潜空间认知推理模型）
**定位**：ChipForge APU 的认知核，<1B 参数，物理 AI 端侧推理
**架构核心**：**Logos**（主线：HRM-Text + GRAM，推理 + 决策） + **SADKO**（探索分支：双向 + FM + FSQ，感知 + 记忆 + 知识 + 多模态）

> 📦 **参考文档体系**（详见各文件）：
> - 总体架构：[`docs/architecture.md`](docs/architecture.md)
> - **Logos 主线白皮书**：[`docs/research/logos-whitepaper.md`](docs/research/logos-whitepaper.md)
> - **Logos K 值策略**：[`docs/research/logos-k-strategy.md`](docs/research/logos-k-strategy.md)
> - **Logos 路线图**：[`docs/research/logos-roadmap.md`](docs/research/logos-roadmap.md)
> - **SADKO 白皮书**：[`docs/research/sadko-whitepaper.md`](docs/research/sadko-whitepaper.md)
> - **SADKO 多模态原生设计**：[`docs/research/sadko-multimodal-native.md`](docs/research/sadko-multimodal-native.md)
> - 外部研究笔记：[`docs/references/README.md`](docs/references/README.md)（HRM-Text、GRAM、TRM、RRM 谱系、LoopCoder-v2、Huginn、STARS、Per-Token Convergence 等）
> - 内部 R&D 提案：[`docs/rfcs/README.md`](docs/rfcs/README.md)（GRR-300M、MR-300M、RDD-0001 K-sweep Plan B 等）
> - 相关项目元信息：[`docs/references/projects.md`](docs/references/projects.md)（ChipForge / HydraForge / AgenticLlama / minimind）

---

## 2. 技术架构（v1.0）

### v1.0 架构（Demo）

```
图像输入
  ↓
原生统一感知层（3层 CNN，~150M） ← 只训这层
  ↓
HRM-Text backbone（~850M）       ← 预训练权重直接用
  ↓
语义图生成（轻量 DiT，~50M）
```

### v1.5 架构（完整方案）

```
图像 / 文本 / 知识图谱
  ↓
原生统一感知层（3层 CNN，~150M）
  ↓
分层递归潜空间引擎（~750M）—— HRM + GRAM
  ├── 高层（慢速）：全局逻辑、长程因果
  ├── 低层（快速）：局部特征、细粒度对齐
  └── 机器潜空间语言（内部 K 次循环，多轨迹采样）
  ↓
双流解码层（~100M）
  ├── 语言解释头（PrefixLM）
  └── 生成式语义头（条件 DiT）
```

---

## 3. 训练阶段

| 阶段 | 内容 | 目标 |
|------|------|------|
| **阶段一** | 潜空间自监督预热 | 让模型学会"机器潜空间语言" |
| **阶段二** | 仅回答目标 + 双任务对齐 | 只对最终输出计算 loss |
| **阶段三** | 多轨迹概率推理 | GRAM stochastic sampling |

---

## 4. 关键技术

### 4.1 HRM-Text 核心
- 递归潜空间推理（K 次内部循环）
- MagicNorm（深层梯度稳定性）
- 40B tokens 训练（千倍少于常规模型）

### 4.2 GRAM 核心
- Stochastic latent trajectories（多假设推理）
- Amortized variational inference
- 测试时 scaling（递归深度 + 并行轨迹）

### 4.3 原生统一感知
- 像素和文字在同一 latent space 联合迭代
- 3 层 CNN 轻量感知层
- 无独立视觉编码器（VE-free）

---

## 5. 开发原则

### 5.1 复用优先
- **HRM-Text 预训练权重**：直接获取使用，不从零训练 LLM
- **现有工具链**：Triton（AgenticLlama）、CppHDL（芯片部署）

### 5.2 阶段验证
- 每个阶段有明确的验证指标
- 先验证感知层集成，再验证双流解码
- 最后验证多轨迹推理

### 5.3 量化方案
- minimind LLM：FP16（不能用 Q4）
- Vision encoder：INT8/Q4
- Action Head：INT8

---

## 6. 里程碑

| 时间 | 里程碑 | 状态 |
|------|--------|:---:|
| +1 个月 | HRM-Text 权重获取 + 感知层集成 | 🔨 进行中 |
| +2 个月 | 感知层微调（冻结 backbone）| ⏳ 待开始 |
| +3 个月 | 双流解码（语言 + 语义图）| ⏳ 待开始 |
| +4-5 个月 | v1.0 Demo 完成 | ⏳ 规划中 |
| +7-8 个月 | GRAM 多轨迹推理集成 | ⏳ 规划中 |
| +10-12 个月 | v1.5 完整训练完毕 | ⏳ 规划中 |

**详细路线图见 [docs/research/logos-roadmap.md](docs/research/logos-roadmap.md)。**

> 📌 **K-sweep 降级说明（2026-07-29）**：LoopCoder-v2 论文已证明循环架构方向正确（K=2 在 10 个 benchmark 一致优于无循环）。K 值问题不再是阻塞性前置研究，而是集成失败的 **Plan B**。触发条件与实验范围详见 [RDD-0001](docs/rfcs/RDD-0001-k-sweep-experiment.md) 与 [logos-k-strategy.md](docs/research/logos-k-strategy.md)。

---

## 7. 研究路线分工（双轨制）— 2026-07-29 战略决策

> **背景**：LoopCoder-v2 论文（2026-06）实证发现 PLT 架构在 K=3 循环崩溃，挑战"循环越多越深"假设。这迫使项目重新审视主线（HRM-Text）的循环深度假设，并明确 SADKO 探索分支的定位。

### 7.1 两条路线的技术偏置与任务分工

| 维度 | 主线（HRM-Text + GRAM）| SADKO 探索分支 |
|------|------------------------|----------------|
| **核心范式** | 分层循环 + 多轨迹采样 | 双向注意力 + Flow Matching + FSQ |
| **数学偏置** | 序列 + 收敛 + 不确定性 | 空间 + 流形 + 离散锚点 |
| **天然优势域** | **推理 + 决策** | **感知 + 记忆 + 知识 + 多模态** |
| **不适合** | 连续流形感知（K 次迭代引入噪声）| 离散序列决策（双向破坏因果）|
| **backbone** | HRM-Text（1.15B 层次化）| AR-Native + Split-GQA + ELF |
| **目标硬件** | ChipForge APU 端侧（<1B 激活）| 服务器训练 + 端侧 KV 缓存 |

### 7.2 主线（Logos）的 K 值策略（不再阻塞）

**2026-07-29 战略调整**：K-sweep 实验从"阻塞性前置研究"降级为"集成失败时的 Plan B"。

**理由**（[logos-k-strategy.md](docs/research/logos-k-strategy.md)）：
1. **架构方向已证**：LoopCoder-v2 在 10 个 benchmark 上证明 K=2 > K=1（+50%），循环架构方向正确
2. **K 是工程调优，非架构判断**：K=2 vs K=4 vs K=8 是任务/规模依赖的工程参数，不需要前置阻塞
3. **HRM-Text 论文自证**：原论文已用 K=8（2H×3L）在 ARC-AGI 验证，无需我们再证

**降级后的 Plan B 触发条件**（仅当集成中出现问题时启动）：
- K=8 推理时延超过 ChipForge APU 预算
- K=8 出现 gain-then-collapse 现象
- 端侧 K 必须 ≤ 2 时

**实验范围收缩**：从 4 实验 × 6 K 值（54h）→ 2 K 值 × 1 任务（<10h）

详细论证见 [logos-k-strategy.md](docs/research/logos-k-strategy.md)。

### 7.3 SADKO 探索分支的定位升级

> **2026-07-29 增补**：SADKO 从"文本认知推理备选"升级为"**多模态流形记忆与感知**"核心研究方向。详细论证见 [SADKO 多模态原生设计](docs/research/sadko-multimodal-native.md)。

**升级依据**（第一性原理）：
- 双向注意力（无因果 Mask）天然处理流形（图像/视频/3D）
- Flow Matching（ODE 可逆变换）天然处理压缩-解压对称
- FSQ 离散化天然支持跨模态共享字典
- 这三者**不是文本架构的扩展，而是连续流形的设计动机**

**与主线融合点**：
- SADKO 训练后冻结的 ELF Memory KV
- 通过 **Cross-Attention** 喂给主线 HRM 的中间层
- "**HRM 是大脑皮层，ELF 是海马体**"

### 7.4 关键原则

1. **不锁定原则**：两条路线均未完成小规模验证前，不锁定任一方向
2. **机制优先**：64M 阶段产出《已验证/已证伪机制清单》比"最佳性能"更重要
3. **融合优先**：两条路线最终应融合，而非互斥（ELF KV → HRM Cross-Attention）
4. **K 自适应**：K 不是固定超参，应根据任务难度 + per-token 收敛状态动态调整

### 7.5 决策时间表

| 时点 | 决策 |
|------|------|
| **+0 月**（立即）| 启动 K-sweep 前置实验（RDD-0001）|
| **+2-3 月** | K-sweep 出结果，决定 HRM-Text K 值 + 是否启用 STARS |
| **+2-3 月**（并行）| SADKO 64M 四大机制实验出《机制清单》|
| **+4 月** | 决定 SADKO 是否升级为正式分支，融合接口是否启动 |
| **+6 月** | 启动 300M 双脑训练（若决策通过）|
| **+12 月** | 1B 端侧推理核 + 1.5B 多模态记忆核 |

---

**版本**：v1.2
**最后更新**：2026-07-29

> 📌 **v1.1 变更**：新增 §7 研究路线分工（双轨制），明确主线 = 推理+决策、SADKO = 感知+记忆+知识+多模态。
> 📌 **v1.2 变更**（2026-07-29）：
> - 主线正式命名为 **Logos / 逻各斯**（希腊哲学：理性之原则）
> - 新增 Logos 文档体系：[logos-whitepaper.md](docs/research/logos-whitepaper.md) / [logos-k-strategy.md](docs/research/logos-k-strategy.md) / [logos-roadmap.md](docs/research/logos-roadmap.md)
> - K-sweep 实验从"阻塞性前置"降级为"Plan B"，详见 [logos-k-strategy.md](docs/research/logos-k-strategy.md)
> - §6 里程碑简化，去掉 K-sweep 阻塞项
> - §7.2 K 值策略调整