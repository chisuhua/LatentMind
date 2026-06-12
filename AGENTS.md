# AGENTS.md — LatentMind 开发手册

## 1. 项目概述

**项目**：LatentMind（原生多模态潜空间认知推理模型）
**定位**：ChipForge APU 的认知核，<1B 参数，物理 AI 端侧推理
**架构核心**：HRM-Text latent space reasoning + GRAM multi-trajectory reasoning

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

| 时间 | 里程碑 |
|------|--------|
| +1 个月 | HRM-Text 权重获取 + 感知层集成 |
| +2 个月 | 感知层微调（冻结 backbone）|
| +3 个月 | 双流解码（语言 + 语义图）|
| +4-5 个月 | v1.0 Demo 完成 |
| +7-8 个月 | GRAM 多轨迹推理集成 |
| +10-12 个月 | v1.5 完整训练完毕 |

---

**版本**：v1.0
**最后更新**：2026-06-12