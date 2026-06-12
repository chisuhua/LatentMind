# 相关项目元信息

> **用途**：为 LatentMind 项目提供上下游依赖、生态系统的快速参考
> **最后更新**：2026-06-13

---

## 1. 上下游关系图

```
            ┌─────────────────────┐
            │   ChipForge APU     │  芯片硬件（部署目标）
            │   (RISC-V+CUDA+TC)  │
            └──────────┬──────────┘
                       │ 部署
                       ▼
            ┌─────────────────────┐
            │     LatentMind      │  ← 本项目（认知核）
            │  （HRM-Text+GRAM）  │
            └──────────┬──────────┘
                       │ 推理结果
                       ▼
            ┌─────────────────────┐
            │     HydraForge      │  推理调度层（下游）
            └──────────┬──────────┘
                       │ 算力调用
                       ▼
            ┌─────────────────────┐
            │   AgenticLlama      │  Triton 推理引擎（提供底层算子）
            └─────────────────────┘

            minimind ─── 平行独立产品线（纯文本认知）
```

---

## 2. 项目卡片

### 2.1 ChipForge（芯片硬件）

| 字段 | 值 |
|---|---|
| **关系** | 部署目标 |
| **定位** | 自研 APU 芯片 |
| **组成** | RISC-V 控制核 + CUDA SIMT 计算核 + Tensor Core 矩阵加速 |
| **本地路径** | `/workspace/project/ChipForge/` |
| **关键约束** | LLM backbone FP16（不能用 Q4）；Vision encoder INT8/Q4；Action Head INT8 |
| **LatentMind 部署位置** | 芯片软件栈**最顶层**（认知核） |

### 2.2 LatentMind（认知核 — 本项目）

| 字段 | 值 |
|---|---|
| **关系** | — |
| **定位** | 物理 AI 时代的原生多模态潜空间认知推理模型 |
| **架构核心** | HRM-Text（确定性 H/L 循环）+ GRAM（多轨迹随机化） |
| **目标规模** | <1B 激活参数，端侧本地推理 |
| **多模态输入** | 图像 / 文本 / 知识图谱（v1.5 完整版） |
| **多模态输出** | 动作 / 语义图 / 自然语言解释 |
| **本地路径** | `/workspace/project/LatentMind/` |
| **技术来源** | [HRM-Text](./hrm-text.md)、GRAM（待补） |
| **当前状态** | 概念设计完成；权重获取进行中；0 行实现代码 |

### 2.3 HydraForge（推理调度层）

| 字段 | 值 |
|---|---|
| **关系** | 下游消费方 |
| **定位** | 多模型推理调度（推测） |
| **本地路径** | `/workspace/project/HydraForge/` |
| **对 LatentMind 价值** | LatentMind 推理结果汇入 HydraForge 进行任务级调度 |
| **待补充** | 具体职责边界、API 形态、SLA 要求 |

### 2.4 AgenticLlama（推理引擎）

| 字段 | 值 |
|---|---|
| **关系** | 底层算子供应商 |
| **定位** | Triton 优化推理引擎 |
| **本地路径** | `/workspace/project/AgenticLlama/` |
| **对 LatentMind 价值** | 提供 HRM-Text 风格循环 / PrefixLM attention 的 Triton kernel |
| **关键技术** | FlashAttention、PrefixLM 双段注意力、静态 KV cache、动态循环退出 |
| **待补充** | 算子覆盖度、量化支持、HRM 风格循环的并行策略 |

### 2.5 minimind（独立产品线）

| 字段 | 值 |
|---|---|
| **关系** | 平行独立产品线（**不是** LatentMind 的子模块） |
| **定位** | 纯文本认知推理（推测） |
| **本地路径** | `/workspace/project/minimind/` |
| **对 LatentMind 价值** | 文本认知能力的"参照系"；可能共享部分数据准备/Tokenizer 工具链 |
| **区别** | 纯文本 vs 多模态；独立训练路径 |

---

## 3. 跨项目依赖矩阵

| 上游 / 下游 | ChipForge | **LatentMind** | HydraForge | AgenticLlama | minimind |
|---|---|---|---|---|---|
| **ChipForge** | — | 部署目标 | — | — | — |
| **LatentMind** | 消费算力 | — | 推理结果 | 算子调用 | 平行 |
| **HydraForge** | — | — | — | 算子调用 | — |
| **AgenticLlama** | — | — | — | — | — |
| **minimind** | 部署目标 | 平行 | — | 算子调用 | — |

---

## 4. 共享/可复用资源

| 资源 | 来源 | 复用方式 |
|---|---|---|
| HRM-Text 预训练权重 | Sapient (HF) | 冻结 backbone + 接入感知层 |
| GRAM 风格随机化 | KAIST/NYU/Mila | 集成到 H 模块 residual |
| Triton 推理算子 | AgenticLlama | PrefixLM attention、循环退出 kernel |
| 数据集标注流水线 | Sapient data_io | 复用 seqpack + stratified sampling |
| 量化工具链 | ChipForge | INT8/Q4 校准、混合精度切分 |

---

## 5. 待补充的研究资料

| 来源 | 优先级 | 状态 |
|---|---|---|
| **HRM-Text**（核心） | 🔴 必做 | ✅ 见 [hrm-text.md](./hrm-text.md) |
| **GRAM**（v1.5 集成） | 🔴 必做 | ⏳ 待收集（OpenReview: Vxu6kcIjwV） |
| HRM 原始 27M（符号版本） | 🟡 参考 | ⏳ 待收集（arXiv:2506.21734） |
| Ouro 1.4B（对比基线） | 🟢 可选 | ⏳ 待收集 |
| Huginn 3.5B（对比基线） | 🟢 可选 | ⏳ 待收集 |
| TRM（小型递归） | 🟢 可选 | ⏳ 待收集 |
| HRM-MoE（社区扩展） | 🟡 备选方案 | ⏳ 待收集 |
| MagicNorm 原始论文 | 🟡 深入理解 | ⏳ 待收集 |
| AdamATan2 优化器 | 🟡 复现细节 | ⏳ 待收集（论文 [20]） |
| FlashAttention 3 | 🟡 部署相关 | ⏳ 通用知识，不专列 |

---

## 6. 文档维护说明

- 本文件维护项目间**关系**与**引用**，不复制各项目内部细节
- 各项目的深入笔记单独成文（`hrm-text.md`、`gram.md` 等）
- 新增相关项目时请追加卡片并更新关系图
- 路径变化时及时更新 `本地路径` 字段

**最后更新**：2026-06-13
