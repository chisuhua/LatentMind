# 四线综合脉络（Logos / SADKO / Hippo / Thumos）

> **一句话定位**：LatentMind 项目四研究线的**总体脉络文档**——将项目既有架构（Logos + SADKO + Hippo）与 V4 议题中"三 Session + Fork/Join + 智能触发 + 知识生成"等讨论内容系统归并到四条研究线，明确每线的核心科学问题、边界、依赖、64M 阶段产出与跨线协调机制。
>
> **性质**：研究线路层面的"骨架索引 + 议题归并"，是后续每线深入展开（白皮书 / 验证计划 / 子方向文档）的统一参考
>
> **最后更新**：2026-07-31
>
> **关键阅读对象**：
> - 想理解"项目有几条研究线、各自做什么"的读者 → 读 §1
> - 想把 V4 议题映射到正确研究线的读者 → 读 §3-§5
> - 想了解 64M 阶段每线核心实验的读者 → 读 §6
> - 想做跨线决策的读者 → 读 §7-§9
> - 想立即动手的读者 → 读 §10

---

## 0. 阅读指南与文档范围

### 0.1 本文不是什么

- ❌ **不是**任何一线的白皮书或详细架构文档（详见各线 README / whitepaper）
- ❌ **不是**实施计划或脚本规范（详见 `docs/implementation/`）
- ❌ **不是**外部研究笔记（详见 `docs/references/`）
- ❌ **不是**内部 R&D 提案（详见 `docs/rfcs/`）

### 0.2 本文是什么

- ✅ 项目研究线路的**总体地图**
- ✅ V4 议题中所有讨论内容（14 议题 + 18 概念 + 18 不确定点）的**系统性归并**
- ✅ 跨线协调的**契约体系与依赖关系**
- ✅ 64M 验证阶段的**每线最小验证实验清单**

### 0.3 与其他文档的关系

```
docs/research/
├── README.md                        ← 顶级研究线索引（指向本文件 + 各线 README）
├── four-lines-synthesis.md          ← 本文（四线综合脉络）
├── REFACTOR_MAP.md                  ← 2026-07-30 重构引用清单（历史）
│
├── logos/                           ← Logos 主线（H/L 循环 AR 主干）
│   ├── README.md                    ← Logos 索引
│   ├── whitepaper.md                ← Logos 白皮书
│   └── ...
│
├── sadko/                           ← SADKO 主干（异构双脑 + 扩散桥梁）
│   ├── README.md
│   ├── whitepaper.md
│   └── ...
│
├── hippo/                           ← Hippo 独立研究线（FM 记忆 + FSQ，原名 ELF）
│   ├── README.md                    ← Hippo 索引（含胼胝体契约）
│   └── ...
│
└── thumos/                          ← Thumos 独立研究线（agent 能力内化为模型循环）[2026-07-31 新增]
    ├── README.md                    ← Thumos 索引（含赫尔墨斯契约）
    └── ...
```

---

## 1. 四线战略定位

### 1.1 四线对照表

| 研究线 | 子目录 | 定位 | 架构核心 | 数学偏置 | 哲学隐喻 |
|--------|--------|------|----------|---------|---------|
| **Logos 主线** | [`logos/`](./logos/) | **推理 + 决策**（端侧哲学）| 分层双时间尺度循环（H/L）+ Per-Token 早退 + 多轨迹并行 | 序列 + 收敛 + 不确定性 | 大脑皮层（理性思考，λόγος）|
| **SADKO 主干** | [`sadko/`](./sadko/) | **感知 + 记忆 + 知识 + 多模态**（流形压缩）| 异构双脑（左 AR + 右 Hippo/FM/FSQ）+ 扩散桥梁（胼胝体）| 空间 + 流形 + 离散锚点 | 海马体 + 皮层协同（Садко 民间叙事）|
| **Hippo 独立研究线** | [`hippo/`](./hippo/) | **右脑关键技术独立验证**（记忆内容 / KG 压缩生长 / FM 检索提取）| 复用 SADKO 右脑机制（双向 + FM + FSQ），独立验证路径 | 同 SADKO | 海马体（ἱππόκαμπος，hippocampus 缩写，原名 ELF）|
| **Thumos 独立研究线** | [`thumos/`](./thumos/) | **agent 能力内化为模型循环动力学**（Intent/Fork-Join/触发/在线-离线作为架构机制）| V4 三 Session 架构作为**循环结构模式**（非外部服务）+ 赫尔墨斯契约与 HydraForge 协同 | 主动执行 + 状态机 + 反馈回路 | 激情/驱动力（θυμός，柏拉图三灵魂之一，位于理性与欲望之间）|

### 1.2 哲学隐喻的自洽性

四条研究线在哲学隐喻上构成完整的"理性认知系统"：

```
柏拉图《理想国》三灵魂（隐喻来源）：
├── logos（理性）       → Logos 主线：推理 + 决策
├── thumos（激情/spirit）→ Thumos 主线：内化 agent 行为，主动执行的驱动力
└── eros（欲望）        → Hippo 主线：记忆渴求 / 联想 / 知识召回

外部补充：
└── 海马体（hippocampus）→ Hippo 主线（基于解剖学隐喻，与 thumos 互补）
```

**命名长度对称性**：Logos (5) / SADKO (5) / Hippo (5) / Thumos (6)——希腊 + 俄罗斯 + 希腊 + 希腊，多元但风格统一。

### 1.3 四线之间的"互补 + 融合"关系

| 关系 | 描述 |
|------|------|
| **Logos ↔ SADKO** | 互补：Logos 推理决策，SADKO 感知记忆。融合接口："Hippo KV → HRM Cross-Attention"（AGENTS.md §7.4）|
| **Logos ↔ Hippo** | 间接消费：Hippo 提供记忆底座 → 通过 SADKO 胼胝体契约 → Logos 读取 |
| **SADKO ↔ Hippo** | 核心 vs 延伸：Hippo 是 SADKO 右脑关键技术的独立研究线，通过胼胝体契约接回 |
| **Logos ↔ Thumos** | 扩展：Thumos 的"内化编排"作用在 Logos 主干之上，将 Logos 从"纯 H/L 推理"扩展为"内含 agent 行为的 H/L 推理" |
| **SADKO ↔ Thumos** | 互补：SADKO 胼胝体管"模型 ↔ 模型"信息流，Thumos 赫尔墨斯契约管"模型 ↔ 外部 agent"信息流 |
| **Hippo ↔ Thumos** | 消费：Thumos 的双层触发 L1 信号可源自 Hippo 的检索置信度（I3 接口返回值）|

---

## 2. 与项目既有架构的整合

### 2.1 四线与既有"双轨 + 探索分支"的关系

**AGENTS.md §7** 定义的双轨战略：
> Logos（主线：分层循环 + 多轨迹推理） + SADKO（探索分支：双向 + FM + FSQ）
> **关键战略**：两条路线都是全新架构，从零训练；64M 是起点，不是验证

**四线是双轨的自然扩展**：

| 时点 | 战略状态 | 变化 |
|------|---------|------|
| 2026-07-29 | 双轨制（Logos + SADKO）| AGENTS.md §7.5 决策时间表 |
| 2026-07-30 | 三线制（+ Hippo）| Hippo 从 SADKO 内部延伸为独立研究线（[REFACTOR_MAP.md](./REFACTOR_MAP.md)）|
| 2026-07-31 | **四线制**（+ Thumos）| Thumos 从 V4 议题中识别为独立科学问题（agent 内化能力）|

**为何不破坏双轨原则**：
- Logos / SADKO **保持原貌**（双轨分工未变）
- Hippo 是 SADKO **内部延伸**（已在 2026-07-30 建立）
- Thumos 是**新维度**（agent 能力内化），不替代任何现有线路

### 2.2 四线与上下游项目的关系

```
            ┌─────────────────────┐
            │   ChipForge APU     │  芯片硬件（部署目标）
            └──────────┬──────────┘
                       │ 部署
                       ▼
            ┌─────────────────────┐
            │     LatentMind      │  ← 本项目（认知核）
            │   四线综合：         │
            │   • Logos（推理决策）│
            │   • SADKO（感知记忆）│
            │   • Hippo（记忆验证）│
            │   • Thumos（内化编排）│
            └──────────┬──────────┘
                       │ 推理结果
                       ▼
            ┌─────────────────────┐
            │     HydraForge      │  推理调度层（外部 agent，与 Thumos 协同）
            └──────────┬──────────┘
                       │ 算力调用
                       ▼
            ┌─────────────────────┐
            │   AgenticLlama      │  Triton 推理引擎（底层算子）
            └─────────────────────┘
```

**关键澄清**（参见 [docs/references/projects.md](../references/projects.md)）：
- **HydraForge**（外部推理调度层）：与 Thumos 通过**赫尔墨斯契约**协同
- **AgenticLlama**（Triton 推理引擎）：为四线提供算子支持
- **minimind**：平行独立产品线，与 LatentMind 无关

### 2.3 四线与 Phase 0 共享前置的关系

**Phase 0**（双轨 64M 共享前置，详见 [logos/sadko-64m-coordination.md §2](./logos/sadko-64m-coordination.md#2-phase-0共享前置验证-5-周)）：
- P.0.1 MiniMind3 64M Dense 从零训练基线
- P.0.2 3 层 CNN 感知层映射
- P.0.3 数据 pipeline + tokenizer
- P.0.4 训练基础设施
- P.0.5 评估框架（统一可比）
- P.0.6 端侧并行化基础设施（Logos 专用）

**Thumos 与 Phase 0 的关系**：
- ✅ 共享 P.0.1 / P.0.3 / P.0.4 / P.0.5（基座、数据、训练、评估）
- ⚠️ P.0.2 感知层：Thumos 部分依赖（Intent 内化可能需要感知层支持）
- ⚠️ P.0.6 端侧并行化：Thumos 强依赖（内化编排的端侧延迟是核心收益）

**待协调**：Thumos 加入后，Phase 0 是否需要扩展 P.0.6 / 新增 P.0.7（赫尔墨斯契约接口实现）→ 详见 §7.1。

---

## 3. V4 议题归并（14 议题 → 四线映射）

> **V4 来源**：会话开始时用户提供的"V4 Final 议题清单"，含 14 项议题（议题 0-13）。本节将所有议题系统性归并到四线。

### 3.1 议题归并总表

| V4 议题 | 主归属 | 副归属 | 理由 |
|---------|--------|--------|------|
| **议题 0**：设计公理与演进逻辑 | **跨线** | — | 所有线路的设计哲学基础 |
| **议题 1**：顶层架构（三路 Session） | Thumos | HydraForge（外化部分）| 三 Session 作为循环结构模式 |
| **议题 2**：Fork/Join 协作机制 | Thumos | SADKO（Hard Join Control Token 借鉴扩散对齐）| 四种 Join 模式作为架构机制 |
| **议题 3**：外部知识库构建 | Hippo | Thumos（知识库类型影响内化决策）| 知识库分层组织属于记忆底座 |
| **议题 4**：ELF 压缩训练 | **Hippo** | — | FM 压缩 = Hippo 核心（直接对应）|
| **议题 5**：ELF 解压 + Query-Aware 解压 | **Hippo** | SADKO（消费侧 Adapter 协调）| 条件化重新生成属于 Hippo 流形操作 |
| **议题 6**：Meta-Key 生成与索引 | **Hippo** | Thumos（Meta-Key 权重受 Intent 调节）| Meta-Key 索引属于 Hippo 检索提取 |
| **议题 7**：Nano-WM 范式 | **Logos** | — | 并行 FFN + 门控属于 AR 主干集成 |
| **议题 8**：联合多层检索生成网络 | **Hippo** | Logos（多层 Hidden State 来自 Logos 主干）| 检索 Query 生成属于 Hippo 方向3 |
| **议题 9**：分层注入策略 | **Logos** | Thumos（路由规则由内化 Intent 决定）| 事实 vs 决策的注入路径属于 AR 消费 |
| **议题 10**：双层动态触发 | **Thumos** | Logos（L1 AR 信号）| L0 意图层 + L1 信号协同属于内化编排 |
| **议题 11**：方言适配器 | **SADKO** | Logos（消费侧接口）| Conditional Adapter 作为胼胝体契约扩展 |
| **议题 12**：端到端训练流水线 | **跨线** | — | 所有线路的训练协调 |
| **议题 13**：工程落地 | **跨线** | — | 显存/调度/可观测性统一 |

### 3.2 归属统计

| 线路 | 议题数（含跨线） | 独占议题 |
|------|:----------------:|----------|
| **Logos** | 2 | 议题 7（Nano-WM）|
| **SADKO** | 1 | 议题 11（方言适配器）|
| **Hippo** | 4 | 议题 4 / 5 / 6 / 8 |
| **Thumos** | 3 | 议题 1 / 2 / 10 |
| **跨线** | 3 | 议题 0 / 12 / 13 |

### 3.3 关键议题详解（每线选 1 个核心议题深入）

#### 议题 4（Hippo 独占）— ELF 压缩训练

**核心问题**：ELF 如何学习将知识压缩为通用 Latent？压缩是"条件化生成"还是"一次性编码"？

**Hippo 视角的关键修正**：
- ❌ V4 原始描述："一次性编码 `encode(x) → latent`"
- ✅ Hippo 视角：**条件化生成** `flow(noise, condition) → latent`（同一知识在不同 Condition 下走出不同轨迹，得到不同 Latent）
- ✅ 离线：学习 Flow 轨迹 + 预计算锚点；在线：根据 AR Query 条件化走完轨迹

**64M 验证实验**：
1. **消费阶梯实验**（最高优先级）：AR + Adapter 输入 (i) GT 压缩嵌入 (ii) FM 生成嵌入 (iii) 无
   - (i)≫(ii) ⇒ ELF 是瓶颈
   - (ii)≈(i)>(iii) ⇒ 验证通过

#### 议题 7（Logos 独占）— Nano-WM 范式

**核心问题**：外部知识如何注入 AR 内部计算而不修改 AR 原始架构？

**Logos 视角的关键修正**：
- ❌ 早期：标准 Cross-Attention 注入（破坏预训练权重）
- ✅ Logos Nano-WM：在 Transformer Layer 添加**与主 FFN 并行的辅助 FFN**，通过 Gate 混合：`output = main_ffn(x) + gate * parallel_ffn(knowledge)`

**64M 验证实验**：
1. **随机注入 vs 真注入对比**（最高优先级）：64M H/L + Nano-WM，注入 (i) 真知识 (ii) 随机向量 (iii) 无
   - (i)≈(ii) ⇒ Gate 不消费知识（机制失效）
   - (i)≫(ii) ⇒ 验证通过

#### 议题 10（Thumos 独占）— 双层动态触发

**核心问题**：何时触发检索？仅 AR 内部 Uncertainty 为什么不够？

**Thumos 视角的关键修正**：
- ❌ V4 原始：Intent Session 是独立物理 Session
- ✅ Thumos 视角：**L0 意图层与 L1 AR 信号都内化为模型 forward pass 内的反馈环路**（Intent 作为模型自身状态机的一部分）

**64M 验证实验**：
1. **内化 vs 外化对比**（最高优先级）：64M 模型三组对比：
   - (i) 内化编排（模型自己的 Intent + 双层触发）
   - (ii) 同一模型 + 外部编排（模拟 HydraForge）
   - (iii) 无编排基线
   - (i)≈(ii)≫(iii) ⇒ 内化不劣于外部
   - (i)≫(ii) ⇒ 内化胜利
   - (i)<(ii) ⇒ 内化假设失败

### 3.4 议题依赖关系图

```
议题0 (设计公理，跨线)
 │
 ├─→ 议题1 (三Session) ──→ 议题2 (Fork/Join) ──→ 议题10 (双层触发)
 │       [Thumos]              [Thumos]              [Thumos]
 │
 ├─→ 议题3 (知识库) ──→ 议题4 (ELF压缩) ──→ 议题5 (ELF解压) ──→ 议题6 (Meta-Key) ──→ 议题8 (联合多层检索)
 │       [Hippo]              [Hippo]              [Hippo]              [Hippo]              [Hippo]
 │
 │                                            ↓
 ├─→ 议题7 (Nano-WM) ←── 议题9 (分层注入) ←── 议题11 (方言适配器)
 │       [Logos]              [Logos]              [SADKO]
 │
 └─→ 议题12 (端到端训练) ──→ 议题13 (工程落地)
         [跨线]                 [跨线]
```

---

## 4. V4 概念归并（18 概念 → 四线映射）

> **V4 来源**：会话中提供的"全部新增/强化概念清单"，含 18 个核心概念（概念 1-18）。本节将所有概念归并到四线。

### 4.1 概念归并总表

| 概念编号 | 概念名 | 主归属 | 副归属 | 关键修正一句话 |
|---------|--------|--------|--------|----------------|
| **1** | ELF 压缩 = 条件化生成 | **Hippo** | — | `flow(noise, condition) → latent`，非 `encode(x) → latent` |
| **2** | ELF 解压 = 条件化重新生成 | **Hippo** | SADKO | 不是回忆原文，是按 AR 需求重新生成变体 |
| **3** | 外部知识三类来源 | **Hippo** | Thumos | 静态/动态/合成，ELF 是知识合成器 |
| **4** | ELF = 知识生成协处理器 | **Hippo** | 跨线 | 核心价值是条件化生成灵活性，非压缩率 |
| **5** | 三个静态陷阱 | **跨线** | — | 单一 Encoder / Linear MLP / 纯语义 Meta-Key 均不可行 |
| **6** | 离线/在线双模式框架 | **跨线** | Thumos | 每个组件的离线/在线行为本质不同 |
| **7** | Conditional Generation Loss | **Hippo** | SADKO | 条件化生成必须优于无条件生成 |
| **8** | Nano-WM FFN 并行 + 门控 | **Logos** | — | 不用 Cross-Attn 注入，用并行 FFN + Gate 混合 |
| **9** | 联合多层检索生成网络 | **Hippo** | Logos | 不用 Per-Layer Per-Query，用多层聚合统一检索 |
| **10** | Query-Aware 按需解压 | **Hippo** | SADKO | 非全量一次性解压，按 Query 动态触发 |
| **11** | 事实 vs 决策分开注入 | **Logos** | Thumos | 事实→门控轻量，决策→Prefill 重量级 |
| **12** | Meta-Key 意图导向 | **Hippo** | Thumos | 从"找相似的词"到"找对当前推理步骤有用的知识" |
| **13** | AR Uncertainty 盲区 | **Logos** | Thumos | "不知道自己不知道"，必须 L0+L1 双层触发 |
| **14** | 四种 Join 模式 | **Thumos** | SADKO | Soft/Hard/Feedback/Fork，各有适用场景和可微性 |
| **15** | Conditional Adapter 替代 MLP | **SADKO** | Logos | 上下文敏感的变形器，非固定翻译器 |
| **16** | 多教师蒸馏（T5+Decoder）| **Hippo** | SADKO | T5 语义 + Decoder 因果，Flow 擦过两个流形 |
| **17** | V1→V2→V3 演进 | **跨线** | — | 结构化动态 = 动态性在结构化容器中运行 |
| **18** | ELF 纯净 + 方言后置 | **Hippo** | SADKO | ELF 不掺杂 AR 特有表征，转换只在 Adapter 中发生 |

### 4.2 归属统计

| 线路 | 概念数 |
|------|:------:|
| Logos | 2 |
| SADKO | 1 |
| Hippo | 9 |
| Thumos | 0 独占 + 5 副归属 |
| 跨线 | 3 |

### 4.3 关键概念详解（每线选 1 个核心概念）

#### 概念 4（Hippo 独占）— ELF = 知识生成协处理器

**关键修正**：ELF 不是"高压缩比编码器"，而是"**与 AR 共生的、可条件控制的知识生成协处理器**"。

**对四线架构的影响**：
- ❌ 评估指标不应只是"压缩率 / 重建误差"
- ✅ 应关注"条件化生成的多样性"和"对 AR 生成质量的提升"
- ✅ 训练目标需加入 Conditional Generation Loss
- ✅ ELF 与 AR 的关系是"共生"而非"主-从"

**对应 Hippo 方向**：方向1（记忆内容架构）+ 方向3（FM 检索提取）联合产出

#### 概念 8（Logos 独占）— Nano-WM FFN 并行 + 门控

**关键修正**：不采用标准 Cross-Attention 注入（在 Self-Attn 和 FFN 之间插入 Cross-Attn 层），因为这会修改 AR 原始架构、增加推理延迟、破坏预训练权重。

**Logos 视角的实现**：
- 在 Transformer Layer 中添加**与主 FFN 并行的辅助 FFN**
- 外部知识作为并行 FFN 的输入/条件
- 通过 Gate `gate ∈ [0, 1]` 混合：`output = main_ffn(x) + gate * parallel_ffn(knowledge)`
- Gate=0 时退化为原始 AR，天然支持降级

**对应 Logos 64M 验证**：v2.0 B.3 依赖 Per-Token 早退 + Nano-WM Gate

#### 概念 14（Thumos 独占）— 四种 Join 模式

**关键修正**：Join 不是一种机制，而是**四种本质不同的信息融合模式**，且都可内化为模型架构机制：

| Join 模式 | 性质 | 可微性 | 适用场景 |
|-----------|------|--------|---------|
| **Soft Join** | Cross-Attention / Nano-WM 门控 | 可微 | 事实性知识（轻量、逐 Token）|
| **Hard Join** | Control Token 插入 | 不可微（REINFORCE / Gumbel-Softmax）| 推理链触发、模式切换（重量级、离散）|
| **Feedback Join** | Uncertainty 反向触发 | 异步事件 | 增量检索（闭环）|
| **Fork Point** | Prefill 后分流 | 异步 | 并行检索不阻塞首 Token |

**Thumos 视角的内化**：
- Soft Join → Logos Nano-WM Gate（已是架构机制）
- Hard Join → 模型词表扩展 + Control Token 嵌入（需 Thumos 验证）
- Feedback Join → 模型 forward pass 内反馈回路（需 Thumos 验证）
- Fork Point → Prefill 后 KV 分流机制（需 Thumos 验证）

---

## 5. V4 不确定点归并（18 ❓ → 四线映射）

> **V4 来源**：会话中提供的"不确定点/开放问题清单"，含 18 个问题（❓1-❓18）。本节将问题分类归属，并区分**架构承重**vs**工程可调**。

### 5.1 不确定点分类（按 V4 描述模式推断）

| ❓ 编号 | 问题摘要 | 主归属 | 承重等级 | 状态 |
|--------|---------|--------|:--------:|------|
| ❓1 | Nano-WM 并行 FFN 每层 vs 选择性层 | Logos | 🟠 承重 | 实验决定 |
| ❓2 | 门控信号（Gate）的具体来源 | Logos | 🟠 承重 | 设计决定 |
| ❓3 | 联合多层检索网络的具体结构 | Hippo | 🟢 可调 | 实验决定 |
| ❓4 | 解压与并行 FFN 的精确先后关系 | SADKO / Logos | 🟠 承重 | 接口契约需锁定 |
| ❓5 | 注入路径划分的中间地带 | Logos / Thumos | 🟠 承重 | 设计决定 |
| ❓6 | 方言适配器与并行 FFN 的精确对接 | SADKO / Logos | 🔴 架构承重 | 训练前必须决定 |
| ❓7 | Nano-WM + Gate 训练方式 | Logos | 🟠 承重 | 设计决定 |
| ❓8 | 多路 KV 显存共享策略 | Thumos | 🟢 可调 | 工程优化 |
| ❓9 | 合成知识质量保障机制 | Hippo | 🟠 承重 | 训练目标 |
| ❓10 | 动态知识在线压缩延迟 | Hippo | 🟢 可调 | 工程优化 |
| ❓11 | 离线锚点存储格式 | Hippo | 🟢 可调 | 工程优化 |
| ❓12 | 多教师蒸馏融合方式 | Hippo | 🔴 架构承重 | 训练前必须决定 |
| ❓13 | Intent Session 架构与训练 | Thumos | 🟠 承重 | 设计决定 |
| ❓14 | Control Token 词表设计 | Thumos | 🟠 承重 | 设计决定 |
| ❓15 | Phase 3 联合微调稳定性 | 跨线 | 🔴 架构承重 | 训练前必须决定 |
| ❓16 | Flow 采样步数权衡曲线 | Hippo | 🟢 可调 | 工程优化 |
| ❓17 | 三路 Session 模型共享底层参数 | Thumos | 🟠 承重 | 设计决定 |
| ❓18 | 意图漂移检测阈值策略 | Thumos | 🟢 可调 | 实验决定 |

### 5.2 承重等级说明

| 等级 | 含义 | 处理时点 |
|------|------|----------|
| 🔴 **架构承重** | 决定整个架构成立与否 | **训练前必须锁定** |
| 🟠 **承重** | 显著影响机制有效性 | 设计阶段需收敛 |
| 🟢 **可调** | 可通过实验 / 工程优化调整 | 训练中持续调优 |

### 5.3 训练前必须解决的 4 个 🔴 承重问题

| ❓ 编号 | 问题 | 必须在 64M 训练前解决的原因 |
|---------|------|----------------------------|
| **❓6** | 方言适配器与并行 FFN 精确对接 | 训练目标 Loss 函数定义必需；否则反向传播路径不通 |
| **❓12** | 多教师蒸馏融合方式 | 决定 ELF 空间几何性质；改变整个 Hippo 训练范式 |
| **❓15** | Phase 3 联合微调稳定性 | 决定能否做端到端联合训练；否则只能冻结分阶段训 |

> ⚠️ 这些问题的答案不在 V4 中提供——**需在 64M 训练启动前补齐**，否则风险极高。

---

## 6. 每线 64M 研究内容

### 6.1 验证哲学（继承自项目）

所有线路共享以下五条验证原则（详见各线 README §验证哲学）：

1. **P1 消融隔离**：每次实验只改变一个机制变量
2. **P2 强制早停**：64M 阶段 30% 训练进度无改善即停
3. **P3 鲁棒性 > 性能**：5+ 随机种子全部通过才算鲁棒
4. **P4 负结果资产化**：失败实验产出《证伪报告》
5. **P5 接口契约层验证**：验证跨线契约可实现

**鲁棒性评分体系**（0-10）：
```
= (机制有效性 × 0.4) + (超参敏感性 × 0.2) + (跨数据集一致性 × 0.2) + (接口契约满足度 × 0.1) + (负结果清晰度 × 0.1)
```

### 6.2 Logos 64M 研究

**核心科学问题**：Nano-WM 门控注入能否在 H/L 循环中**真正被使用**（gate 开启），且 K≤4 预算下保留收敛性？

**机制清单（产出目标）**：

| 机制 | 验证方式 | 通过标准 |
|------|---------|---------|
| Nano-WM Gate 行为 | 注入 (i) 真知识 (ii) 随机 (iii) 无；测量 gate 熵分布 | (i)≫(ii)≫(iii) |
| K 预算下收敛稳定性 | MagicNorm + Per-Token 早退集成 | PPL 退化 <0.2 vs MiniMind3 基线 |
| 知识 vs 随机注入 delta | 合成 KV-recall 任务（含干扰项）| 真注入 acc > 随机 +20% |
| Gate 坍缩检测 | gate 熵在 5+ 种子的分布 | 不坍缩到 ~0 / ~0.5 |

**Riskiest Bet 实验**：随机注入 vs 真注入对比（单次实验信息密度最高）

**新增验证章节（2026-07-31 同步）**：
- **§3.5 表征对齐探针（DiscoLoop）**：H/L 循环 OOD 多跳是否面临表征错位（cosine < 0.5 ID / < 0.4 OOD）→ 决定是否引入 Φ 通道
- **§3.6 Loop B 备选实验（条件启动）**：StreamingLLM / InfLLM / FM+FSQ 三档端侧 fallback
- **§4.6 L-PLT 评估（条件启动）**：L 循环完整 PLT position shift，端侧延迟 -33%
- **§4.6.9 L-PLT + Φ 联合实验**：延迟 + 质量联合优化（Q.Φ 共享 §3.5 Φ 模块）

**参考文档**：
- [logos/whitepaper.md §2.2](./logos/whitepaper.md#22-模块-blogos-分层递归潜空间引擎-750m)
- [logos/64m-validation-plan.md](./logos/64m-validation-plan.md)（v1.5：§3.5 + §3.6 + §4.6 + §4.6.9）
- [logos/k-strategy.md](./logos/k-strategy.md)（v1.5：§2.2.7 PLT + Φ 集成设计）
- [logos/sadko-64m-coordination.md](./logos/sadko-64m-coordination.md)
- [logos/codebook-config-sop.md](./logos/codebook-config-sop.md)（FSQ [8,8,4] 配置 SOP，L.3 备选前置）
- [logos/fsq-consumption-design.md](./logos/fsq-consumption-design.md)（Logos 消费 Hippo FSQ 条件设计）

### 6.3 SADKO 64M 研究

**核心科学问题**：**"右脑冻结、左脑只读"** 的契约能否在 64M 用**小 Adapter** 学到？还是必须联合训练？

**机制清单（产出目标）**：

| 机制 | 验证方式 | 通过标准 |
|------|---------|---------|
| Conditional Adapter 可训性 | 冻结 ELF + Adapter vs 联合训练 | Adapter-only ≳ joint - ε |
| 多教师蒸馏（T5 + Decoder）| 分别冻结 / 联合 / 加权 三组对比 | 联合优于单一 T5 或 Decoder |
| 胼胝体契约五元组（I1-I5）| 端到端验证 | INV-1 ~ INV-5 全部满足 |
| 扩散对齐有效性 | v3.0 四大实验（详见 SADKO 64M 计划）| A/B/C/D/E 全部通过 |

**Riskiest Bet 实验**：冻结 + Adapter vs 联合训练（直接验证 scaling 经济学）

**参考文档**：
- [sadko/whitepaper.md](./sadko/whitepaper.md)
- [sadko/64m-validation-plan.md](./sadko/64m-validation-plan.md)
- [sadko/v1-architecture.md](./sadko/v1-architecture.md)
- [sadko/v2-architecture.md](./sadko/v2-architecture.md)
- [sadko/v3-architecture.md](./sadko/v3-architecture.md)
- [sadko/multimodal-native.md](./sadko/multimodal-native.md)

### 6.4 Hippo 64M 研究

**核心科学问题**：FM 能否**生成**（不仅是重建）落在流形上的知识 Embedding，且被独立训练的 AR 通过小 Adapter 消费后产生正向 delta？

**机制清单（产出目标）**：

| 机制 | 验证方式 | 通过标准 |
|------|---------|---------|
| ELF 生成保真度 | 消费阶梯实验：(i) GT (ii) FM 生成 (iii) 无 | (ii)≈(i)>(iii) |
| 条件化生成多样性 | 同一知识在不同 Condition 下的 Latent 距离 | 距离 > ε（不是确定性映射）|
| on-manifold 率 | FM 样本与真实 manifold 的 k-NN 距离分布 | > 80% 在 ε-邻域内 |
| FSQ codebook 使用 | Code 使用率 / 坍缩检测 | 使用率 > 80%，无明显坍缩 |
| Meta-Key 检索 Recall | 意图导向 vs 语义导向 | 意图导向 Recall > 语义导向 +10% |

**Riskiest Bet 实验**：消费阶梯实验（GT/FM/无）—— 一次性验证 ELF 生成声明 + Adapter 可训性 + Nano-WM 消费能力

**参考文档**：
- [hippo/README.md](./hippo/README.md)（含胼胝体契约）
- [hippo/memory-architecture.md](./hippo/memory-architecture.md)
- [hippo/graph-growth.md](./hippo/graph-growth.md)
- [hippo/retrieval-extraction.md](./hippo/retrieval-extraction.md)

### 6.5 Thumos 64M 研究

**核心科学问题**：agent 风格的意图解析、动态知识访问、自触发 re-retrieval 能否**内化为模型本身的循环动力学**，且在相同参数下**优于外部编排**？

**机制清单（产出目标）**：

| 机制 | 验证方式 | 通过标准 |
|------|---------|---------|
| 内化 Intent 解析 | (i) 内化 (ii) 外部 Intent Session (iii) 无 | (i)≈(ii)≫(iii) 或 (i)≫(ii) |
| 内化 Fork/Join | 在多步推理任务上对比 | 同上 |
| 双层触发内化反馈 | L0 ↔ L1 反馈环路时序验证 | 反馈延迟 < 50ms |
| 离线/在线内部动力学 | 训练 vs 推理模式差异可识别 | 模式分类准确率 > 95% |
| 赫尔墨斯契约六元组 | 端到端验证 | H1-H6 + INV-1~6 全部满足 |
| INV-3 降级保证 | 内化失败时 HydraForge 接管 | 接管成功率 100% |

**Riskiest Bet 实验**：内化 vs 外部编排 vs 无编排三组对比（必须含完全离线模式，保护"内化"声明不被 HydraForge 撑场）

**参考文档**：
- [thumos/README.md](./thumos/README.md)（含赫尔墨斯契约）
- [thumos/intent-internalization.md](./thumos/intent-internalization.md)（待建）
- [thumos/fork-join-internalization.md](./thumos/fork-join-internalization.md)（待建）
- [thumos/offline-online-dynamics.md](./thumos/offline-online-dynamics.md)（待建）

### 6.6 64M 阶段的最高优先级三组实验（跨线）

根据信息密度 + 风险敏感度，以下三组实验应**并行启动**：

| 优先级 | 实验 | 线路 | 验证目标 | 假阴性风险 |
|:------:|------|------|---------|-----------|
| **🥇** | **消费阶梯实验** | Hippo | ELF 生成 + Adapter + Nano-WM 消费 | 中（含 GT 对照）|
| **🥈** | **随机 vs 真注入对比** | Logos | Nano-WM Gate 是否真消费知识 | 低（含随机对照）|
| **🥉** | **内化 vs 外部 vs 无编排** | Thumos | agent 能力是否可内化 | 中（必须含离线模式）|

---

## 7. 跨线协调与契约体系

### 7.1 现有协调机制

| 协调文档 / 契约 | 范围 | 状态 |
|----------------|------|------|
| [logos/sadko-64m-coordination.md](./logos/sadko-64m-coordination.md) | Logos ↔ SADKO 双轨协调（Phase 0/1/2/3）| ✅ 已建立 |
| Hippo 胼胝体契约（[hippo/README.md §2](./hippo/README.md#2-胼胝体接口契约corpus-callosum-contract)）| Hippo ↔ SADKO 模型-模型信息流 | ✅ 已建立（含 I6 Memory Token Interface）|
| Thumos 赫尔墨斯契约（[thumos/README.md §2](./thumos/README.md#2-赫尔墨斯接口契约hermes-contract)）| Thumos ↔ HydraForge 模型-外部信息流 | ✅ 本文档新增（2026-07-31）|
| [logos/codebook-config-sop.md](./logos/codebook-config-sop.md) | Logos ↔ Hippo FSQ 配置协调（[8,8,4] vs 256）| ✅ 2026-07-31 新增 |
| [logos/fsq-consumption-design.md](./logos/fsq-consumption-design.md) | Logos ↔ Hippo FSQ 消费条件设计（4 前置门控 + INV-6~9）| ✅ 2026-07-31 新增 |
| [../research/loop-memory-survey.md](./loop-memory-survey.md) | 循环+记忆系统文献谱系（Loop A/B/C 分类）| ✅ 2026-07-31 新增 |

### 7.2 待新增的协调条目

| 协调 | 待协调内容 | 负责线路 | 时点 |
|------|----------|---------|------|
| Thumos 加入 Phase 0 | P.0.2 / P.0.6 是否扩展 / 新增 P.0.7 | Logos / Thumos / 协调中枢 | T+1（Thumos 验证前）|
| 胼胝体契约扩展 | I6 Memory Token Interface 数值化（维度/形状/范围）| Hippo / Thumos | T+1（INV-6 待 SADKO owner 签字）|
| 四线 Phase 2 交叉验证 | 现有 Phase 2 仅 Logos vs SADKO，需扩展为四线对比 | Logos / SADKO / Hippo / Thumos | +5 月 |
| 四线决策矩阵 | 现有双轨决策矩阵需扩展为四线 | 协调中枢 | +6 月 |
| Loop B fallback 应用 | §3.6 三档备选（StreamingLLM / InfLLM / FM+FSQ）的端侧评估协调 | Logos / Hippo | §3.5 探针后 |

### 7.3 契约体系全景图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          外部世界                                     │
│   HydraForge（外部 agent）     AgenticLlama（Triton 引擎）            │
└─────────────┬────────────────────────────┬──────────────────────────┘
              │ H1-H6                       │ 算子 API
              │ 赫尔墨斯契约                 │
              ▼                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Thumos（内化 agent）                              │
│  Intent + Fork/Join + 双层触发 + 离线/在线动力学（内化为循环）         │
└──────────┬─────────────────────────────────────┬─────────────────────┘
           │ 消费                                │ 消费
           ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────────────────┐
│  Logos 主线          │              │  Hippo 独立研究线                │
│  H/L 循环 AR 主干    │              │  FM 压缩/解压 + FSQ + Meta-Key   │
│  + Nano-WM Gate      │              │  + 检索 API                       │
└──────────┬───────────┘              └──────────┬───────────────────────┘
           │ Cross-Attention (AGENTS.md §7.4)     │ I1-I5
           │ "Hippo KV → HRM"                       │ 胼胝体契约
           └─────────────┬─────────────────────────┘
                         ▼
              ┌──────────────────────┐
              │  SADKO 主干          │
              │  异构双脑 + 扩散桥梁  │
              │  Conditional Adapter │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  ChipForge APU       │
              │  （端侧部署目标）     │
              └──────────────────────┘
```

---

## 8. 演进路径

### 8.1 短中期（2026-07 至 +3 月）：四线 64M 并行验证

```
+0 月 ──┬─ Phase 0（共享前置，5 周）
        │  ├─ P.0.1 - P.0.6（已有）
        │  └─ P.0.7（Thumos 加入后新增，2 周）
        ├─ Phase 1A Logos 64M（v1.0 / v2.0 / v3.0 三阶段）
        ├─ Phase 1B SADKO 64M（v1.0 / v2.0 / v3.0 三阶段）
        ├─ Phase 1C Hippo 三方向（独立）
        └─ Phase 1D Thumos 三方向（独立）
        
+2 月 ── 四线 64M 验证报告（机制清单）
        ├─ Logos：Nano-WM 机制清单 + K 值推荐
        ├─ SADKO：四大实验结果 + 已验证/已证伪机制清单
        ├─ Hippo：消费阶梯 + 检索验证
        └─ Thumos：内化 vs 外部 vs 无编排对比
```

### 8.2 中期（+3 月至 +6 月）：四线决策与升格

```
+3 月 ── 决策门：哪条线升格 300M 训练？
        ├─ Logos 推理强 + 知识注入有效 → 升格 300M
        ├─ SADKO 记忆强 + 灵魂注入有效 → 升格 300M
        ├─ Hippo 检索生成保真 → 加入 SADKO 胼胝体
        └─ Thumos 内化编排有效 → 与 HydraForge 协同（INV-3 兜底）

+5 月 ── Phase 2 交叉验证（四线对比）
        ├─ 6 类任务：推理 / 代码 / 决策 / 记忆 / 端侧 / 多模态
        └─ 输出《四线对比报告》

+6 月 ── Phase 3 决策矩阵
        ├─ 四线并行 / 三线并行 / 单线升格 / 退回通用
        └─ 启动 300M 训练（决策后）
```

### 8.3 远期（+9 月至 +12 月）：规模化与融合

```
+9 月  ── Logos 1B 端侧 Demo（独立）
+12 月 ── Logos 1B + SADKO 1.5B 完整融合
         ├─ Cross-Attention 融合接口
         ├─ Thumos 内化编排完整实现
         └─ HydraForge 外部调度协同
```

---

## 9. 验证哲学与决策标准

### 9.1 64M 阶段的核心判断矩阵

| 维度 | 验证通过 | 验证失败 |
|------|---------|---------|
| **机制有效性** | 主指标达标率 ≥ 90% | < 70%（标记"已证伪"）|
| **超参敏感性** | 1.0（不敏感）| < 0.5（极度敏感）|
| **跨数据集一致性** | 全数据集通过 | 单一数据集通过（标记"过拟合"）|
| **接口契约满足度** | 五/六元组全部可实现 | 任一不变量违反（标记"接口失稳"）|
| **负结果清晰度** | 失败案例有清晰诊断 | 失败原因不明（标记"未充分消融"）|

### 9.2 跨线决策的关键时点

| 时点 | 决策 | 决策依据 |
|------|------|---------|
| +0.5 月 | 启动 Phase 0 | P.0.1 MiniMind3 基线 |
| +1.5 月 | Phase 0 完成 | 6-7 项全部通过 |
| +2 月 | 启动 Phase 1（四线并行） | 四线各自机制清单 |
| +5 月 | Phase 2 交叉验证 | 四线对比 |
| +6 月 | Phase 3 决策矩阵 | 四线并行 / 三线 / 单线 / 退回 |
| +6+ 月 | 启动 300M | 决策后启动 |

---

## 10. 立即执行清单

### 10.1 文档层（已部分完成）

- [x] **本次任务（v0.1, 2026-07-31）**：创建 `docs/research/thumos/README.md` + `docs/research/four-lines-synthesis.md`
- [x] **本次任务**：更新 `docs/research/README.md` 顶级索引，添加 Thumos 入口
- [x] **本次任务**：创建 `docs/research/thumos/intent-internalization.md` 骨架
- [x] **本次任务**：创建 `docs/research/thumos/fork-join-internalization.md` 骨架
- [x] **本次任务**：创建 `docs/research/thumos/offline-online-dynamics.md` 骨架
- [ ] 更新 `logos/sadko-64m-coordination.md` 加入 Thumos 协调条目（或新建 `thumos/thumos-64m-coordination.md`）

### 10.2 设计层（Phase 0 启动前必须解决）

- [ ] 解决 ❓6（方言适配器与并行 FFN 精确对接）
- [ ] 解决 ❓12（多教师蒸馏融合方式）
- [ ] 解决 ❓15（Phase 3 联合微调稳定性）

### 10.3 实验层（64M 验证最高优先级三组）

- [ ] **Hippo 消费阶梯实验**（GT/FM/无 + Adapter 三组）
- [ ] **Logos 随机 vs 真注入对比**（含 Gate 熵分布测量）
- [ ] **Thumos 内化 vs 外部 vs 无编排对比**（含完全离线模式）

### 10.4 协调层（Phase 0 启动前）

- [ ] Thumos 加入 Phase 0 的协调决议
- [ ] 赫尔墨斯契约与 HydraForge owner 的首次对齐会议
- [ ] 胼胝体契约是否扩展 I6 的决议

---

## 11. 相关文档索引

### 11.1 四线入口

- **Logos 主线**：[logos/README.md](./logos/README.md) / [logos/whitepaper.md](./logos/whitepaper.md)
- **SADKO 主干**：[sadko/README.md](./sadko/README.md) / [sadko/whitepaper.md](./sadko/whitepaper.md)
- **Hippo 独立研究线**：[hippo/README.md](./hippo/README.md)
- **Thumos 独立研究线**：[thumos/README.md](./thumos/README.md)

### 11.2 跨线协调文档

- **双轨协调中枢**：[logos/sadko-64m-coordination.md](./logos/sadko-64m-coordination.md)
- **Phase 0 实施指南**（外部）：[`../implementation/phase-0-implementation-guide.md`](../implementation/phase-0-implementation-guide.md)
- **Phase 0 失败回退 SOP**（外部）：[`../implementation/phase-0-recovery-sop.md`](../implementation/phase-0-recovery-sop.md)

### 11.3 项目级文档

- **AGENTS.md**：[`../../AGENTS.md`](../../AGENTS.md)（项目开发手册 + 双轨战略）
- **架构总览**：[`../architecture/architecture.md`](../architecture/architecture.md)
- **实施文档索引**：[`../implementation/README.md`](../implementation/README.md)
- **外部参考**：[`../references/README.md`](../references/README.md)
- **R&D 提案**：[`../rfcs/README.md`](../rfcs/README.md)
- **上下游项目卡片**：[`../references/projects.md`](../references/projects.md)

### 11.4 关键参考

- **LoopCoder-v2**：[`../references/loopcoder-v2.md`](../references/loopcoder-v2.md)（K=2 > K=1 +50%）
- **STARS**：[`../references/stars.md`](../references/stars.md)（循环崩溃根因 + Jacobian 修复）
- **Per-Token Convergence**：[`../references/per-token-convergence.md`](../references/per-token-convergence.md)（90% token 6 步收敛）
- **HRM-Text**：[`../references/hrm-text.md`](../references/hrm-text.md)（架构灵感参考）

---

## 12. 文档历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-07-31 | 初创。整合 V4 议题清单（14 项）+ 概念清单（18 项）+ 不确定点清单（18 项）到四线（Logos/SADKO/Hippo/Thumos）。新增 Thumos 第四研究线（agent 能力内化为模型循环动力学）。建立赫尔墨斯契约（H1-H6 + INV-1~6）与 HydraForge 协同。 |

---

**最后更新**：2026-07-31
**版本**：v0.1
**作者**：Sisyphus（综合 V4 议题清单与项目既有文档的脉络梳理）