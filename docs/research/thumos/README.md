# Thumos 独立研究线（Thumos Research Line，第四研究线）

> **定位**：LatentMind 项目第四研究线，专门研究"**智能体能力内化为模型循环动力学**"——将 V4 议题中的 Intent Session / Fork/Join / 双层触发 / 离线在线双模式 / 三 Session 编排等**原本属于 Agent 的能力**，作为**模型自身的循环结构模式**实现（而非外部调度），与 HydraForge（外部推理调度层 / AgenticLlama）形成"**内化 + 外化协同**"的双层架构。
> **状态**：🆕 2026-07-31 启动
> **命名来源**：希腊哲学 Thumos（θυμός）——柏拉图《理想国》三灵魂之一（logos 理性 / thumos 激情 / eros 欲望），位于理性与欲望之间，是"主动执行的驱动力 / spiritedness"。与 Logos（理性）、Hippo（海马体记忆）形成哲学隐喻上的自洽分工。

---

## 0. 定位声明

### 0.1 为什么需要第四条研究线

V4 议题（`docs/research/four-lines-synthesis.md`）中"**三 Session 架构 + Fork/Join + 智能触发**"表面上像 HydraForge 的工作（推理调度层），但**深度审视**后发现：这些机制不是外部调度，而是改写**模型内部如何持续推理**——意图解析、动态知识访问、自触发 re-retrieval 等行为，正在成为**模型循环动力学的一部分**，而非外挂服务。

**核心论证**（参见 [four-lines-synthesis.md §0](../four-lines-synthesis.md)）：
- **科学层面**：System 1/System 2、Global Workspace、Predictive Processing 等认知科学框架都在问"agent 行为能否内化为认知动力学"——独立科学假设，非工程分层
- **架构层面**：若成立，端侧推理的延迟优势是数量级的（无需外部 round-trip）；若不成立，则应老实走"模型 + 外部 RAG"路线——非平凡二选一
- **与 HydraForge 共演化**：内化能力 + 外部调度不是替代而是分工；本线路研究"两者如何协同设计"

### 0.2 与现有三线的关系

| 现有线路 | 本线路关系 |
|---------|----------|
| **Logos 主线**（H/L 循环 AR 主干） | Thumos 的"内化编排"作用在 Logos 主干之上，将 Logos 的循环动力学从"纯 H/L 推理"扩展为"内含 agent 行为的 H/L 推理" |
| **SADKO 主干**（异构双脑 + 扩散桥梁） | Thumos 与 SADKO 协同：SADKO 的"胼胝体契约"管"模型 ↔ 模型"信息流，Thumos 的"赫尔墨斯契约"管"模型 ↔ 外部 agent"信息流 |
| **Hippo 独立研究线**（FM 记忆 + FSQ） | Thumos 消费 Hippo 的检索输出；Thumos 的双层触发 L1 信号可源自 Hippo 的检索置信度（I3 接口返回值） |

### 0.3 与 HydraForge / AgenticLlama 的边界

```
┌────────────────────────────────────────────────────────────────┐
│  HydraForge（外部推理调度层，AgenticLlama 实现）                  │
│  ─ 职责 ─                                                       │
│  • 多模型编排（Logos / SADKO / 第三方 LLM）                      │
│  • 任务级调度（planning / tool use / memory routing）            │
│  • 端到端 SLA 管理（延迟 / 吞吐 / 降级）                          │
│  ─ 边界 ─                                                       │
│  • 持有 session-level 状态（多轮对话 / 工具栈 / 用户偏好）         │
│  • 与模型之间通过明文 API / 协议交互                              │
└─────────────────────────────┬──────────────────────────────────┘
                              │ 赫尔墨斯契约（Hermes Contract）
                              │   内化接口 ↔ 外化接口
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Thumos（模型内 agent 能力，第四研究线，本线路）                   │
│  ─ 职责 ─                                                       │
│  • 模型 forward pass 内的 Intent 解析（L0 意图层）                 │
│  • 模型 forward pass 内的 Fork/Join 编排原语                      │
│  • 模型 forward pass 内的双层触发反馈（L0 ↔ L1）                  │
│  • 离线 / 在线模式作为模型自身状态机                               │
│  ─ 边界 ─                                                       │
│  • 持有 token-level / step-level 状态（KV 缓存 / Hidden State）  │
│  • 与外部 agent 通过契约式数据交换（不是 RPC 调用）                │
└─────────────────────────────┬──────────────────────────────────┘
                              │ 消费 Logos 主干 + Hippo 检索
                              ▼
                Logos 主线 + SADKO/Hippo 记忆
```

---

## 1. 三个并行研究方向

| 方向 | 文件 | 核心问题 |
|------|------|---------|
| **方向1：意图与触发内化**（Intent + Trigger Internalization） | [intent-internalization.md](./intent-internalization.md)（⏳ 待建）| L0 意图解析与 L1 AR 信号的协同能否内化为模型 forward pass 内的反馈环路，替代外部分类器 + 外部不确定性监控？ |
| **方向2：编排原语内化**（Fork/Join Internalization） | [fork-join-internalization.md](./fork-join-internalization.md)（⏳ 待建）| Soft/Hard/Feedback/Fork 四种 Join 模式能否作为模型架构机制（前向计算图 + Gate + Control Token），而非服务层调度决策？ |
| **方向3：离线/在线内部动力学**（Offline/Online Internal Dynamics） | [offline-online-dynamics.md](./offline-online-dynamics.md)（⏳ 待建）| "训练阶段学习 + 推理阶段应用"的离线/在线双模式差异能否内化为模型自身状态机（不是服务部署模式）？ |

**并行性**：三方向最小验证单元可完全独立。

---

## 2. 赫尔墨斯接口契约（Hermes Contract）

> **核心定义**：赫尔墨斯（Hermes / Ἑρμῆς）是希腊神使——边界、过渡、跨界中介之神。本契约定义 **Thumos 线路（内化 agent）与 HydraForge（外部 agent）之间的标准信息交换接口**——既是"内化能力的对外表达"，也是"外化能力的内化锚点"。
>
> **命名理由**：Thumos 是内部 spirited drive，Hermes 是外部 messenger——两者在哲学上互补（内在动力 vs 外部传导），对应本线路与 HydraForge 的协同关系。

### 2.1 设计原则

1. **最小化**：只规定 Thumos ↔ HydraForge 必要的解耦面，不约束内部研究空间
2. **对称性**：内化接口（Thumos 暴露给 HydraForge 的"内化能力快照"）与外化接口（HydraForge 暴露给 Thumos 的"外化能力锚点"）形态对称
3. **可验证**：每个 Thumos 研究成果都能映射到契约中的某个槽位
4. **降级保证**：Thumos 内化失败时，HydraForge 必须能接管所有 agent 能力（关键不变量 INV-3）

### 2.2 接口契约六元组（骨架）

| 接口维度 | 类型 | 输入 | 输出 | 演进约束 |
|---------|------|------|------|---------|
| **H1. Intent Snapshot** | 张量接口 | 模型 forward pass 内 L0 意图解析输出 | `(batch, intent_dim)` 形状的意图向量 + Functional Tags | intent_dim 与 HydraForge 的 planner 对齐 |
| **H2. Join Mode Vector** | 张量接口 | 当前 step 选择的 Join 模式 + Gate 值 | `(batch, 4)` one-hot + `(batch,)` gate scalar | 四种模式必须存在；gate 值 ∈ [0, 1] |
| **H3. Trigger Signal** | 异步事件 | L0 / L1 触发的检索请求 | `(event_type, payload)` 二元组 | 触发必须可降级为"无触发" |
| **H4. Internal State Export** | 训练时检查点 | 模型循环动力学内部状态 | KV Cache + Hidden State + Intent Vector | 仅训练/调试时使用；推理时通过 H1-H3 暴露 |
| **H5. External Capability Anchor** | 协议层 | HydraForge 暴露的工具/数据能力描述 | 函数签名 + Schema | 描述语言与外部 API 解耦 |
| **H6. Failure Handoff** | 降级协议 | Thumos 内化失败 / HydraForge 调用失败 | 静默降级路径（具体值由实现决定）| 必须支持静默降级，不抛异常 |

### 2.3 关键不变量

无论 Thumos 内部如何演进，以下不变量必须保持：

- **INV-1** Intent Snapshot 输出张量形状满足 H1 约束（HydraForge planner 可读取）
- **INV-2** Join Mode Vector 中 Gate 值 ∈ [0, 1]，且 Gate=0 时退化为原始 Logos 行为
- **INV-3** **Thumos 内化失败时，HydraForge 必须能 100% 接管所有 agent 能力**（关键！保证内化是"增益"而非"必需"）
- **INV-4** Trigger Signal 必含失败回退路径（不触发 = 合法状态）
- **INV-5** Internal State Export 路径**默认关闭**，需显式开启（防止训练状态泄漏）
- **INV-6** 接口契约的**变更需双侧 review**（Thumos owner + HydraForge owner 共同签字）

### 2.4 演进路径

| 阶段 | 契约状态 | Thumos 状态 | HydraForge 协同方式 |
|------|---------|------------|--------------------|
| **T0（当前）** | 骨架契约（§2.2 + §2.3）| 三个方向最小验证单元启动 | 完全独立（仅 INV-3 兜底）|
| **T+1** | 数值化（维度/形状/范围）| Thumos 内部机制初步验证 | HydraForge 接收 Intent Snapshot 软引用 |
| **T+2** | 接口实现 reference code | Thumos 三方向产出 reference impl | HydraForge 可选消费 Join Mode Vector |
| **T+3** | 接口锁定（v1.0 contract）| Thumos 三大方向都通过 64M 验证 | HydraForge 强制消费（INV-3 兜底开启）|
| **T+4** | 接口演进（v1.x → v2.0）| Thumos 进入规模化（300M+）| 完整双向协同 |

### 2.5 与现有契约的协同

| 现有契约 | 关系 |
|---------|------|
| **Hippo 胼胝体契约**（I1-I5 + INV-1~5）| 平行关系。Hippo 契约管"模型 ↔ 模型"信息流；Hermes 契约管"模型 ↔ 外部 agent"信息流。两者在 H3 (Trigger Signal) 处有交集——Trigger 可源自 Hippo 的检索置信度（I3 Retrieval API 返回值）。 |
| **SADKO 胼胝体对齐**（白皮书 §2.3）| Thumos 的 Join Mode Vector 中 Hard Join（Control Token）可能复用 SADKO 的扩散对齐信号 |

### 2.6 外部研究参考 — DiscoLoop 哲学共鸣声明（2026-07-31 增补）

> 📌 **保守声明**：本节为**哲学立场登记**，**不是** Thumos 设计验证证据。

**DiscoLoop 论文**（[Fu et al., arXiv:2607.00341, 2026](https://arxiv.org/abs/2607.00341)；详见 [docs/references/discoloop.md](../../references/discoloop.md)）的核心立场是"**外部推理能力内化为循环结构**"——让多跳推理不需要外显 CoT，直接在循环内部完成。

**Thumos 与 DiscoLoop 的哲学共鸣点**：
- DiscoLoop："多跳推理内化为循环结构模式"（让外部 CoT 变成循环内部 latent computation）
- Thumos："agent 能力内化为模型循环动力学"（让外部 agent 协调变成循环内部 intent/fork-join/trigger）

两者共享同一哲学立场——**原本属于外部的能力，可成为模型自身的循环结构模式**。

**重要澄清**：
- ✅ Thumos 与 DiscoLoop 在"**内化能力**"哲学方向上**共鸣**
- ✅ DiscoLoop 提供了"内化机制存在可行路径"的间接先例
- ❌ DiscoLoop **不构成** Thumos agent 内化假设的**直接验证证据**
  - DiscoLoop 研究对象：latent multi-hop composition（潜空间内的多步推理组合）
  - Thumos 研究对象：tools / sessions / handoff protocols（外部 agent 协同）
- ❌ DiscoLoop **不证明** Intent/Fork-Join/Trigger 等具体机制的可行性

**对 Thumos 64M 验证计划的影响**：
- Thumos 64M 三大方向验证（[four-lines-synthesis §6.5](../four-lines-synthesis.md#65-thumos-64m-研究)）仍是 Thumos 设计独立证伪/验证的**唯一依据**
- DiscoLoop 提供的仅是"内化哲学的间接先例"，**不改变** 内化 vs 外化对比实验的判定线
- 若 64M 内化 vs 外化对比显示内化不优于外化，**不能**用 DiscoLoop 哲学共鸣作为保留 Thumos 假设的理由

**Thumos 借鉴层级判定**（Oracle 评审）：
- **架构**：独立设计，**不借鉴** DiscoLoop 具体实现
- **哲学**：共鸣，可**共享**"内化能力"立场
- **验证**：独立，**不可互证**

---

---

## 3. 验证哲学（继承项目原则）

### 3.1 核心立场

> **Thumos 独立研究线的目标不是产出"最佳性能"，而是产出"鲁棒性评分"与"负结果清单"**。
> 每个机制的"证伪"与"验证"具有同等战略价值。

### 3.2 五大原则

| # | 原则 | Thumos 应用 |
|---|------|------------|
| **P1** | **消融隔离** | 每个方向的实验**只改变一个机制变量**（如：内化 Intent vs 外部分类器），其他保持固定 |
| **P2** | **强制早停** | 64M 阶段任一指标在 30% 训练进度无改善 → 立即停止，标记"未达阈值" |
| **P3** | **鲁棒性 > 性能** | 同一机制在 5+ 随机种子下"全部通过"才算鲁棒；单一种子通过不计入 |
| **P4** | **负结果资产化** | 每个失败实验产出《证伪报告》：失败原因 + 是容量问题还是架构问题 |
| **P5** | **接口契约层验证** | 除机制本身外，必须验证 §2 赫尔墨斯契约的六元组可实现，**特别是 INV-3 降级保证**（内化失败 → HydraForge 接管）|

### 3.3 鲁棒性评分体系（0-10）

```
鲁棒性评分 =
    (机制有效性 × 0.4)          // 主指标达标率
  + (超参敏感性 × 0.2)          // 1.0 = 不敏感，0.0 = 极度敏感
  + (跨数据集一致性 × 0.2)      // 1.0 = 全数据集通过，0.0 = 单一数据集通过
  + (接口契约满足度 × 0.1)       // §2 六元组是否完整可实现
  + (负结果清晰度 × 0.1)         // 失败案例是否有清晰诊断
```

**判定阈值**（继承自 [hippo/README.md §3.3](../hippo/README.md#33-鲁棒性评分体系-0-10)）：
- 评分 ≥ 8.0：可进入 HydraForge 强制协同（T+3 阶段）
- 评分 6.0-7.9：需 300M 中等规模二次验证
- 评分 4.0-5.9：需优化超参 + 多种子验证
- 评分 < 4.0：标记"已证伪"，转入《负结果清单》

---

## 4. 与 V4 三 Session 架构的关系

V4 议题中"三 Session + Fork/Join + 智能触发"在不同视角下的归属：

| 视角 | 归属 |
|------|------|
| 作为**服务部署模式**（三个独立进程，IPC 通信） | HydraForge 职责 |
| 作为**循环结构模式**（一个前向传播内的三个功能阶段） | **Thumos 职责** |
| 作为**混合实现**（内化部分 + 外化部分协同） | Thumos ↔ HydraForge 共同职责 |

**澄清**：V4 的三 Session 架构**既是 HydraForge 的蓝图，也是 Thumos 的目标架构**——同一架构的不同实现层。这正是"内化 + 外化协同"的物理基础。

---

## 5. 文件清单与状态

| 文件 | 状态 | 议题 | Owner |
|------|------|------|-------|
| [README.md](./README.md) | ✅ 已建立 | 索引 + 赫尔墨斯契约 + 哲学声明 | 待指派 |
| [intent-internalization.md](./intent-internalization.md) | ✅ 骨架 | 方向1：L0/L1 意图与触发内化 | 待指派 |
| [fork-join-internalization.md](./fork-join-internalization.md) | ✅ 骨架 | 方向2：Fork/Join 编排原语内化 | 待指派 |
| [offline-online-dynamics.md](./offline-online-dynamics.md) | ✅ 骨架 | 方向3：离线/在线内部动力学 | 待指派 |
| [negative-results.md](./negative-results.md) | ⏳ 待建 | 《负结果清单》| 待指派 |

---

## 6. 维护规范

1. **新增研究记录**：单主题一个文件，并在本 README §5 登记
2. **接口契约变更**：必须遵守 §2.3 INV-6（双侧 review：Thumos owner + HydraForge owner）
3. **失败实验**：必须登记到 `negative-results.md`，附诊断（容量/架构/工程化/泛化/接口不变量违反）
4. **跨文档引用**：使用相对路径（`./`、`../logos/`、`../sadko/`、`../hippo/`、`../four-lines-synthesis.md`）
5. **语言**：中文为主，英文技术术语保留

---

## 7. 相关文档

| 文档 | 关系 |
|------|------|
| [docs/research/four-lines-synthesis.md](../four-lines-synthesis.md) | 四研究线综合脉络（含 V4 议题归并）|
| [docs/research/README.md](../README.md) | 顶级研究线索引（含 Thumos 入口）|
| [docs/research/hippo/README.md](../hippo/README.md) | Hippo 独立研究线（含胼胝体契约参考）|
| [docs/research/sadko/whitepaper.md](../sadko/whitepaper.md) | SADKO 异构双脑 + 扩散桥梁 |
| [docs/research/logos/whitepaper.md](../logos/whitepaper.md) | Logos 主线 H/L 循环 |
| [docs/references/discoloop.md](../../references/discoloop.md) | **DiscoLoop 论文笔记**（§2.6 哲学共鸣参考）|
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 双轨分工战略（Thumos 是其扩展）|
| [docs/references/projects.md](../../references/projects.md) | HydraForge / AgenticLlama 项目卡片 |
| [docs/implementation/phase-0-implementation-guide.md](../../implementation/phase-0-implementation-guide.md) | Phase 0 共享前置（Thumos 也需复用）|

---

**下一步**：
1. 起草三个方向（intent-internalization / fork-join-internalization / offline-online-dynamics）的最小验证单元
2. 与 HydraForge owner 协商赫尔墨斯契约的数值化（T+1 阶段）
3. 更新 [logos/sadko-64m-coordination.md](../logos/sadko-64m-coordination.md) 增加 Thumos 协调条目（或新建 [thumos-64m-coordination.md](./thumos-64m-coordination.md)）

**最后更新**：2026-07-31
**版本**：v0.1（骨架级）