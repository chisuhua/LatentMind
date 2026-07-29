# RFC / 设计提案索引（docs/rfcs/）

> **用途**：集中存放 LatentMind 项目的**内部 R&D 设计提案**（Request For Comments）
> **最后更新**：2026-06-13

---

## ⚠️ 区分 references/ 与 rfcs/

| 目录 | 内容 | 状态 | 来源 |
|---|---|---|---|
| **[docs/references/](../references/)** | **已有外部研究/项目**的笔记（HRM-Text, GRAM, TRM, HRM-原始 27M, RRM 谱系等）| ✅ 已发表 / 已开源 | 外部论文 + 官方仓库 |
| **[docs/rfcs/](./)** | **内部 R&D 设计提案**（自创架构、未实现的工程方案）| 📝 草案 | 内部讨论 / 实验前设计 |

**核心区别**：
- `references/` 记录"**别人做了什么**"——有外部论文/代码支撑
- `rfcs/` 记录"**我们想做什么**"——尚未实现，待评审

---

## 1. 当前 RFC 列表

| RFC | 目标规模 | 状态 | 关联 |
|---|---|---|---|
| [grr-300-gated-recursive-refiner.md](./grr-300-gated-recursive-refiner.md) | 300M（**即插即用插件**）| 📝 草案 | 与 [mr-300](./mr-300-micro-refiner.md) 共享设计哲学 |
| [mr-300-micro-refiner.md](./mr-300-micro-refiner.md) | ~295M（**科学实验平台**）| 📝 草案 | 与 [grr-300](./grr-300-gated-recursive-refiner.md) 关联 |
| [RDD-0001-k-sweep-experiment.md](./RDD-0001-k-sweep-experiment.md) | **MiniMind3 64M（快速消融）**| 📝 **Plan B 草案** | **Logos 主线集成失败的 Plan B**——2026-07-29 由 LoopCoder-v2 论文触发，原为主线前置，**现降级为条件性消融** |

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

---

**最后更新**：2026-06-13
