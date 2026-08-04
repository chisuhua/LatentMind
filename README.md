# LatentMind — 原生多模态潜空间认知推理模型

> **定位**：物理 AI 时代的原生多模态认知推理芯片核心模型
> **架构核心**：**Logos**（主线：分层循环 + 多轨迹推理） + **SADKO**（探索分支：双向 + FM + FSQ） + **Hippo**（独立研究线：记忆 / KG / 检索） + **Thumos**（独立研究线：agent 内化为循环动力学）
> **关键战略**：**全部全新架构，从零训练**——不复用 HRM-Text / GRAM / 等任何外部预训练权重（仅作架构灵感参考）
> **目标**：<1B 激活参数，端侧本地推理，原生视觉+文本+知识图谱联合推理

---

## 核心架构

```
图像 / 文本 / 知识图谱
  ↓
原生统一感知层（3层 CNN，~150M）← 从零训练
  ↓
Logos 分层递归潜空间引擎（~750M）← 从零训练（H/L 双时间尺度）
  ├── H 慢速：全局逻辑、长程因果
  ├── L 快速：局部特征、细粒度对齐
  └── 机器潜空间语言（内部 K 次循环）
  ↓
双流解码层（~100M）← 从零训练
  ├── 语言解释头（PrefixLM）
  └── 生成式语义头（轻量 DiT / Flow Matching）
  ↓
动作 / 语义图 / 自然语言解释
```

> **辅助架构**（独立研究线）：
> - **Hippo**：记忆内容架构 / KG 压缩生长 / FM 检索提取（FSQ 256 离散码本）
> - **Thumos**：Intent / Fork-Join / 触发 / 离线在线 内部化为循环结构模式（替代外部 agent 调度）

---

## 与 ChipForge APU 的关系

LatentMind 是 ChipForge APU 的**认知推理核**，运行在芯片上。

```
ChipForge APU
├── RISC-V 控制核
├── CUDA SIMT 计算核
├── Tensor Core 矩阵加速
└── LatentMind 认知核（软件栈最顶层）
    ├── 原生感知层（CNN，INT8 推理）
    ├── 潜空间引擎（Logos，FP16 推理）
    └── 双流解码器（INT8 推理）
```

---

## 项目结构

```
LatentMind/
├── AGENTS.md            # 项目开发手册（v1.4 战略）
├── README.md            # 本文件（项目总览）
│
├── src/                 # 模型代码（待实现）
│
├── docs/                 # 技术文档
│   ├── architecture/
│   │   └── architecture.md              # 总体技术架构（v1.2）
│   ├── implementation/                 # 近期可执行的工程规范
│   │   ├── README.md
│   │   ├── phase-0-implementation-guide.md   # Phase 0 实施指南
│   │   └── phase-0-recovery-sop.md          # Phase 0 失败回退 SOP
│   ├── references/                      # 外部研究笔记
│   │   ├── README.md                    # 索引
│   │   ├── papers/                       # 论文源文件
│   │   ├── hrm-text.md                  # HRM-Text 笔记
│   │   ├── gram.md                      # GRAM 笔记
│   │   ├── hrm-original.md              # HRM 原始 27M 笔记
│   │   ├── trm.md                       # TRM 笔记
│   │   ├── huginn.md                    # Huginn 3.5B 笔记
│   │   ├── stars.md                     # STARS 2026 笔记
│   │   ├── per-token-convergence.md     # Per-Token 笔记
│   │   ├── loopcoder-v2.md              # PLT 笔记
│   │   ├── discoloop.md                 # DiscoLoop 笔记（Loop A 表征对齐）
│   │   ├── memorizing-transformers.md   # Loop B kNN 检索
│   │   ├── compressive-transformers.md  # Loop B 1D Conv 压缩
│   │   ├── streaming-llm.md             # Loop B 滑动窗口 + sink
│   │   ├── inf-llm.md                   # Loop B 块级 memory
│   │   ├── auto-compressors.md          # Loop B LLM 自压缩
│   │   ├── rmt.md                       # Loop B 特殊 [mem] tokens
│   │   ├── landmark-attention.md        # Loop B attention 内生检索
│   │   ├── rrm-survey.md                # RRM-B 谱系调查
│   │   ├── rrm-reward.md                # RRM-A 笔记
│   │   └── projects.md                  # 相关项目元信息（ChipForge / HydraForge / AgenticLlama / minimind）
│   ├── research/                        # 四研究线核心
│   │   ├── README.md                    # 顶级研究线索引（四线 + 综合 + 调研）
│   │   ├── four-lines-synthesis.md      # 四线综合脉络（V4 议题归并 + 64M 研究内容）
│   │   ├── loop-memory-survey.md        # 循环+记忆系统文献谱系（Loop A/B/C + 7 篇 Loop B）
│   │   ├── REFACTOR_MAP.md
│   │   ├── logos/                       # Logos 主线
│   │   │   ├── README.md
│   │   │   ├── whitepaper.md            # Logos 架构白皮书
│   │   │   ├── k-strategy.md            # K 值策略（含 L-PLT + Φ 集成设计）
│   │   │   ├── 64m-validation-plan.md   # 64M 验证计划（含 §3.5 DiscoLoop 探针 + §3.6 Loop B 备选 + §4.6 L-PLT）
│   │   │   ├── sadko-64m-coordination.md # Logos/SADKO 双轨协调中枢
│   │   │   ├── codebook-config-sop.md   # FSQ [8,8,4] vs 256 冲突 SOP
│   │   │   ├── fsq-consumption-design.md # Logos 消费 Hippo FSQ 条件设计
│   │   │   ├── v1-architecture.md
│   │   │   ├── v2-architecture.md
│   │   │   ├── v3-architecture.md
│   │   │   └── roadmap.md
│   │   ├── sadko/                       # SADKO 探索分支
│   │   │   ├── README.md
│   │   │   ├── whitepaper.md
│   │   │   ├── multimodal-native.md
│   │   │   ├── open-issues.md          # 包含 B-09 Compressive 1D Conv 备选
│   │   │   ├── 64m-validation-plan.md
│   │   │   ├── v1-architecture.md
│   │   │   ├── v2-architecture.md
│   │   │   ├── v3-architecture.md
│   │   │   └── hippo-*.md               # 右脑 Hippo 历史文档
│   │   ├── hippo/                       # Hippo 独立研究线
│   │   │   ├── README.md                # 含胼胝体接口契约（I1-I6 + INV-1-6）
│   │   │   ├── memory-architecture.md
│   │   │   ├── graph-growth.md
│   │   │   └── retrieval-extraction.md
│   │   └── thumos/                      # Thumos 独立研究线（2026-07-31 启动）
│   │       ├── README.md                # 含赫尔墨斯接口契约（H1-H6 + INV-1-6）
│   │       ├── intent-internalization.md
│   │       ├── fork-join-internalization.md
│   │       └── offline-online-dynamics.md
│   ├── rfcs/                            # 内部 R&D 提案（草案）
│   │   ├── README.md
│   │   ├── grr-300-gated-recursive-refiner.md
│   │   └── mr-300-micro-refiner.md
│   └── superpowers/                      # 内部 spec 文档
│
├── tests/                # 测试
└── examples/             # 示例
```

> 📦 **目录功能区分**：
> - `docs/references/`：**外部研究**笔记（HRM-Text, GRAM, Loop B 7 篇等）
> - `docs/research/`：**项目四研究线**（Logos / SADKO / Hippo / Thumos + 综合 + 调研）
> - `docs/implementation/`：**可执行**工程规范（Phase 0 实施指南 + 失败回退 SOP）
> - `docs/rfcs/`：**内部 R&D 提案**（待评审）
> - `docs/architecture/`：**总体架构**设计

---

## 四研究线战略（2026-07-29 战略决策 + 2026-07-31 增补）

| 研究线 | 定位 | 架构 | 状态 |
|--------|------|------|------|
| **Logos 主线** | 推理 + 决策 + 端侧哲学 | H/L 双时间尺度 + Nano-WM 门控 + Per-Token 早退 + L-PLT 可选 | ✅ 2026-07-29 升级 |
| **SADKO 主干** | 感知 + 记忆 + 知识 + 多模态 | 异构双脑（AR + Hippo）+ 扩散桥梁 + Conditional Adapter | ✅ 2026-07-29 升级 |
| **Hippo 独立研究线** | 记忆 / KG / 检索 | FSQ 256 离散码本 + FM 压缩 + Retrieval API | ✅ 2026-07-30 启动 |
| **Thumos 独立研究线** | agent 内化为循环动力学 | V4 三 Session 作为循环结构模式 + 赫尔墨斯契约 | 🆕 2026-07-31 启动 |

**关键关系**：
- **Logos + SADKO** = 项目双轨分工（详见 [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策)）
- **Hippo** = SADKO 右脑关键技术的独立延伸
- **Thumos** = V4 议题（智能体能力内化）的独立研究线，与 HydraForge 协同

详见 [docs/research/four-lines-synthesis.md](docs/research/four-lines-synthesis.md)

---

## 关键技术来源（**仅作架构灵感参考**）

### 4.1 循环架构（Loop A）
- **HRM-Text**（Sapient，2026-05）：1.15B 参数，40B tokens，latent space 推理
- **GRAM**（Bengio + KAIST + Mila，2026）：生成式递归推理，多轨迹概率推理
- **TRM**（Samsung SAIL，2025-10）：极简递归，K=3 在 ARC-AGI 45%
- **Huginn 3.5B**（2025-02）：50 步循环反例
- **LoopCoder-v2 / PLT**（2026-06）：CLP + G-SWA 并行循环
- **STARS 2026**：循环崩溃修复（Jacobian 谱半径）
- **Per-Token Convergence**（2026-07）：90% token 6 步收敛
- **DiscoLoop**（UC Berkeley + Princeton，2026-07）：表征错位根因 + Φ 通道修复

### 4.2 持久记忆系统（Loop B，2026-07-31 完整覆盖）
- **Memorizing Transformers**（Google，ICLR 2022）：kNN 增强注意力，262K tokens 外部记忆
- **Compressive Transformers**（DeepMind，ICLR 2020）：1D Conv 压缩 + 双粒度 memory
- **StreamingLLM**（MIT + Meta，ICLR 2024）：attention sink + 滑动窗口，4M tokens
- **InfLLM**（清华 + MIT + Meta，NeurIPS 2024）：块级 memory + 训练无关 1M+ tokens
- **AutoCompressors**（Princeton NLP，EMNLP 2023）：LLM 自生成 summary vectors
- **RMT**（MIPT + AIRI，NeurIPS 2022）：特殊 [mem] tokens + BPTT 跨段
- **Landmark Attention**（EPFL，NeurIPS 2023）：attention 内生 block retrieval

### 4.3 多模态原生设计
- **商汤 NEO-Unify**（SenseNova U1 思想）：像素和文字同空间

### 4.4 关键原则
- **从零训练**：Logos / SADKO / Hippo / Thumos 全部新架构，不复用任何预训练权重
- **架构灵感参考 ≠ 集成**：HRM-Text / GRAM / Loop B 论文仅作参考，不集成其训练流程或权重
- **64M 是起点**：MiniMind3 64M Dense 仅作基座参考，逐步扩展 64M → 300M → 1B → 1.5B
- **K 是超参数**：由 64M 验证决定，不是架构决策

详见 [docs/research/loop-memory-survey.md](docs/research/loop-memory-survey.md)（完整 Loop A/B/C 谱系 + 6 大 Loop B 流派 + 对项目启示）

---

## 开发状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| **四研究线立项** | ✅ 完成 | Logos/SADKO 2026-07-29 升级；Hippo 2026-07-30 启动；Thumos 2026-07-31 启动 |
| **64M 验证计划** | 📋 制定完成 | 详见 [docs/research/logos/64m-validation-plan.md](docs/research/logos/64m-validation-plan.md)（v1.5：含 §3.5 DiscoLoop 探针 + §3.6 Loop B 备选 + §4.6 L-PLT）|
| **Phase 0 实施** | ⏳ 待启动 | 6 项 P.0.x 共享前置（详见 [docs/implementation/phase-0-implementation-guide.md](docs/implementation/phase-0-implementation-guide.md)）|
| **64M 训练启动** | ⏳ 待启动 | Logos + SADKO 并行（单卡 RTX 3090 分时段）|
| **v1.0 Demo** | ⏳ 规划中 | +6 月目标 |
| **v1.5 完整方案** | ⏳ 规划中 | +12 月目标 |

详见 [docs/research/logos/roadmap.md](docs/research/logos/roadmap.md)

---

## 相关项目

| 项目 | 关系 |
|------|------|
| **Logos** | 主线架构（项目核心）|
| **SADKO** | 探索分支（与 Logos 双轨）|
| **Hippo** | 独立研究线（SADKO 右脑技术延伸）|
| **Thumos** | 独立研究线（agent 内化）|
| **HydraForge** | 外部推理调度层（LatentMind 下游，与 Thumos 协同）|
| **minimind** | 文本认知推理（独立产品线）|
| **ChipForge** | 芯片硬件（LatentMind 部署目标）|
| **AgenticLlama** | 推理引擎（Triton 优化）|

详见 [docs/references/projects.md](docs/references/projects.md)

---

**最后更新**：2026-07-31（与四研究线 + 64M 验证计划 + Loop B 谱系同步）