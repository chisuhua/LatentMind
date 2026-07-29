# RFC: HRM-Text 循环深度 K-sweep 前置实验（RDD-0001）

> **状态**：❌ **完全废弃**（2026-07-29 v1.2 — **不再需要此 RFC**）
> **作者**：来自工作流（LoopCoder-v2 论文冲击波分析）

> ⚠️ **2026-07-29 重大变更**：由于 Logos / SADKO 都是**全新架构、从零训练**，K 值是**超参数**，本 RFC 已**完全失去意义**，作为历史归档保留。

> **归档原因**：
> 1. **不复用 HRM-Text 权重**——K-sweep 的目的是"找最优 K 用于复刻 HRM-Text"，不复用则失去意义
> 2. **K 值是超参数，不是架构决策**——不需要前置实验"找最佳 K"
> 3. **64M 是起点**——K 值由 64M 验证自然产出，不是预先确定
> 4. **GR AM 不集成**——v3.0 中的"GRAM 多轨迹"被独立实现

> **保留此 RFC 仅供历史参考**。

> **替代方案**：
> - K 值策略详见 [logos-k-strategy.md](../research/logos-k-strategy.md)
> - 64M 验证详见 [logos-64m-validation-plan.md](../research/logos-64m-validation-plan.md)
> - v2.0 多种循环策略详见 [logos-v2-architecture.md §3.1](../research/logos-v2-architecture.md#31-b1-串行-k-循环基线)

---

## 1. 摘要

### 1.1 历史背景

本 RFC 最初是为了"验证 HRM-Text 的 K=4-8 假设是否在 64M 上成立"。2026-07-29 战略调整后：

- ❌ Logos 不复用 HRM-Text 权重
- ❌ K 值不再需要"前置验证"
- ❌ 64M 是 Logos 新架构的**起点**，不是验证阶段
- ✅ 本 RFC 内容已**合并到** [logos-v2-architecture.md §3.1](../research/logos-v2-architecture.md)（多种循环策略对比 §B.1 串行 K 循环基线）

### 1.2 为什么废弃

| 废弃原因 | 详细 |
|---------|------|
| **架构策略变更** | Logos 是全新架构，从零训练，不需要复用 HRM-Text 权重 |
| **K 值定位变更** | K 是超参数，不是固定最优值，不需要前置 K-sweep |
| **64M 定位变更** | 64M 是起点，逐步扩展到 300M / 1B / 1.5B |
| **GR AM 不集成** | 不集成 GRAM 训练流程，多轨迹独立实现 |

### 1.3 替代文档

- [logos-k-strategy.md](../research/logos-k-strategy.md) — K 值作为超参数的新定位
- [logos-v2-architecture.md](../research/logos-v2-architecture.md) — 多种循环策略对比（§B.1 串行 K 循环）
- [logos-v3-architecture.md](../research/logos-v3-architecture.md) — 多轨迹并行（独立实现）
- [logos-64m-validation-plan.md](../research/logos-64m-validation-plan.md) — 64M 起点验证计划

如需更新 K 策略设计，请编辑上述新文档，**不要编辑此 RFC**。

---

**最后更新**：2026-07-29（v1.2 完全废弃）
**作者**：来自工作流（LoopCoder-v2 论文冲击波分析）
**状态**：❌ 完全废弃（仅供历史参考）