# RFC / 设计提案索引（docs/rfcs/）

> **用途**：集中存放 LatentMind 项目的**内部 R&D 设计提案**（Request For Comments）
> **最后更新**：2026-07-31（与 4 研究线 + 64M 验证计划同步）

---

## ⚠️ 三类文档目录的精确区分

| 目录 | 内容 | 状态 | 来源 | 时间范围 |
|---|---|---|---|---|
| **[docs/references/](../references/)** | **外部已有研究/项目**的笔记（HRM-Text, GRAM, TRM, RRM 谱系, DiscoLoop, Loop B 7 篇等）| ✅ 已发表 / 已开源 | 外部论文 + 官方仓库 | 长期参考 |
| **[docs/research/](../research/)** | **项目四研究线**的架构设计、白皮书、验证计划、契约（Logos / SADKO / Hippo / Thumos）| 🔄 持续演进 | 内部设计 | +0 至 +12 月 |
| **[docs/rfcs/](./)** | **跨研究线的实验性设计提案**（自创架构、未实现的工程方案）| 📝 草案 | 内部讨论 | 中期探索 |

**核心区别**：
- `references/` 记录"**别人做了什么**"——有外部论文/代码支撑
- `research/` 记录"**主线在做什么**"——已通过架构评审，纳入项目核心研究
- `rfcs/` 记录"**我们想尝试什么**"——尚未评审，可能被合并到 `research/` 或否决

**RFC 流转路径**：
```
📝 RFC 草案 (rfcs/)
   ↓ 评审通过
   ↓
✅ RFC 实验中 (rfcs/ + src/ 中代码)
   ↓ 证据链完整
   ↓
📐 纳入研究线 (research/logos 或 research/sadko 等)
   ↓
🎯 实施 (implementation/)
```

---

## 1. 当前 RFC 列表

| RFC | 目标规模 | 状态 | 关联 |
|---|---|---|---|
| [grr-300-gated-recursive-refiner.md](./grr-300-gated-recursive-refiner.md) | 300M（**即插即用插件**）| 📝 草案 | 与 [mr-300](./mr-300-micro-refiner.md) 共享设计哲学 |
| [mr-300-micro-refiner.md](./mr-300-micro-refiner.md) | ~295M（**科学实验平台**）| 📝 草案 | 与 [grr-300](./grr-300-gated-recursive-refiner.md) 关联 |
| [RDD-0001-k-sweep-experiment.md](./RDD-0001-k-sweep-experiment.md) | ~~MiniMind3 64M~~ | ❌ **完全废弃**（2026-07-29）| Logos/SADKO 都是全新架构从零训练，K 值是超参数，不再需要此 RFC |

**说明**：当前 RFC 列表相对 64M 验证计划阶段**保持稳定**——GRR/MR 主要是 300M 规模的即插即用替代方案，与 Logos 64M 三阶段验证**正交**。Logos 64M 验证本身不依赖任何 RFC。

---

## 2. RFC 状态机

```
📝 草案 (Draft) — 当前所有 RFC 的状态
   ↓
🔬 实验中 (Experimenting) — 已有 100M/300M 消融代码
   ↓
✅ 已实现 (Implemented) — 集成到 LatentMind 主分支
   ↓
❌ 已否决 (Rejected) — 明确记录否决原因
```

**与 64M 验证计划的关系**：
- 64M 阶段**不直接调用**任何 RFC（独立从零训练）
- 64M 机制清单产出后，**可能**产出新 RFC（如"FSQ 码本扩展为分形码本"等）
- RFC 优先级低于 research/ 中的核心研究

---

## 3. 现有 RFC 摘要

### 3.1 GRR-300M（Gated Recursive Refiner）

- **目标**：300M 即插即用插件，嵌入 ≥7B 基座，提供可观测的推理质量提升
- **核心创新**：
  - Gated SRB（LSTM 思想 → Transformer 原生实现）
  - Confidence Probe + Token-Level Router
  - Adaptive Reset（坏盆地逃逸）
- **关键设计取舍**：300M 不足以支撑双时间尺度分层循环，GRR 用门控+路由实现等效功能
- **额外 FLOPs 上限**：≤ 30-50% 基座

### 3.2 MR-300M（Micro-Refiner）

- **目标**：~295M 验证型实验平台，验证"块内隐式循环 + 自适应深度"假设
- **核心组件**：Gated DeltaNet + GQA + Attention Residuals
- **两阶段训练**：
  - 阶段一：非循环预训练（T=1）建立语义锚点
  - 阶段二：循环微调（T=2→8 渐进展开）
- **核心原则**：仅当 300M 消融证据链完整闭合后，才推进到 1.5B-3B

### 3.3 RDD-0001（已废弃）

- ❌ 2026-07-29 完全废弃
- 原因：Logos/SADKO 都是全新架构从零训练，K 值是超参数，不需要先行的 K-sweep 实验
- 教训：外部架构的实验设计不应凌驾于项目自身架构决策

---

## 4. RFC 维护规范

### 4.1 添加新 RFC

1. 命名：`<代号>-<规模>-<技术名>.md`（例：grr-300-gated-recursive-refiner.md）
2. 必须包含头元信息（状态、作者、目标规模、目标场景、关联文档）
3. 必须明确标注"未实现" / "未验证"
4. 自创组件必须明确指出"无外部论文支撑"
5. 添加到本 README 的 RFC 列表中

### 4.2 推进 RFC 状态

- 草案 → 实验中：需要在 `src/` 下有对应代码
- 实验中 → 已实现：需要在 `AGENTS.md` / `README.md` 路线图中记录
- 任何阶段 → 已否决：必须在 RFC 头部说明否决原因
- **新流程**：实验性 RFC 验证通过后，迁移到 `research/` 对应研究线（如 [research/logos/](../research/logos/) 或 [research/sadko/](../research/sadko/)）

### 4.3 引用 RFC

在代码或文档中引用 RFC 时使用相对路径：
```markdown
见 [grr-300 RFC](../rfcs/grr-300-gated-recursive-refiner.md)
```

---

## 5. 与 references/ 的交叉引用

- [GRR-300M 与 HRM-Text 对比](../references/rrm-survey.md#2-下一代架构的更广泛探索)
- [MR-300M 与 minimind 项目](../session-memory.md)（参考 minimind 数据集与基线）
- [GRR/MR 的 RRM 谱系背景](../references/rrm-survey.md)
- [DiscoLoop 表征对齐](../references/discoloop.md) — Loop A 对齐机制（与 RFC 思路无直接关系）

---

## 6. 与 research/ 的关系

| 类型 | 关系 |
|------|------|
| **research/logos** | RFC 中的循环架构方案（GRR-300M / MR-300M）若评审通过，可能并入 Logos 子方向 |
| **research/sadko** | 异构双脑相关 RFC（暂无）会归入 SADKO |
| **research/hippo** | 记忆 / KG / 检索相关 RFC 归入 Hippo 三个子方向 |
| **research/thumos** | agent 内化相关 RFC 会归入 Thumos 三方向 |

**关键澄清**：RFC 是"**实验性**"机制，research/ 是"**已纳入核心研究**"机制。两者不可混用。

---

**最后更新**：2026-07-31（与 4 研究线 + 64M 验证计划同步）
