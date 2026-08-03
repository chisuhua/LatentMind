# FSQ 代码本配置冲突解决 SOP

> **一句话定位**：解决 SADKO/Hippo FSQ 配置中 `[8,8,8,8]` vs `codebook_size=256` 的冲突，确保 64M 训练配置统一，并为 Logos 后续消费 FSQ 提供稳定接口
> **性质**：跨线协调 SOP（影响 Logos → Hippo 集成路径）
> **最后更新**：2026-07-31
> **上游文档**：[../sadko/open-issues.md §二 B-05](../sadko/open-issues.md)（冲突原始描述）
> **下游影响**：[fsq-consumption-design.md](./fsq-consumption-design.md)（Logos 消费 FSQ 条件设计）

---

## 1. 冲突描述

来自 [open-issues.md §二 B-05](../sadko/open-issues.md)：

| 文档位置 | 配置 | 实际组合空间 | 与 codebook_size 关系 |
|----------|------|:---:|------|
| [validation-plan §4.2](../sadko/64m-validation-plan.md) | `levels=[8,8,8,8]`, `codebook=256` | **4096** | 需 clamp 到 256，浪费 94% 组合 |
| [v3-arch §3](../sadko/v3-architecture.md)（主推）| `levels=[8,8,8,8]` | **4096** | 同上，浪费 94% |
| [v3-arch §3](../sadko/v3-architecture.md)（备选）| `levels=[8,8,4]` | **256** | **完全匹配**，无浪费 |

### 1.1 冲突根因

- **算术**：4 个标量量化维度（levels）= `8^4 = 4096` 个组合码字
- **clamp 后果**：64M 实际只能索引 0-255 → 4096 → 256 强制截断，**94% 的码字空间被抛弃**
- **同码冲突**：clamp 让大量不同原始码字映射到同一索引 → 检索 Top-K 准确率下降
- **耦合影响**：B-02（code_embed 模块）也因此选择 256×768，独立于 v1.0 的 16 条 semantic_tag embedding

### 1.2 为什么 v3-arch 主推 [8,8,8,8] 仍然存在

推测原因：作者可能希望"为未来 4096 码本预留升级空间"。但 64M 强约束下不验证 4096 码本利用率——**这是设计动机与验证约束的错位**。

---

## 2. 解决方案

### 2.1 统一裁决（推荐）

**64M 阶段统一为 `[8,8,4] = 256`**：
- ✅ 完全匹配 `codebook_size=256`，无 clamp 浪费
- ✅ 码本利用率目标 > 80% 可达成（参考 [v3-arch §6 实验 A](../sadko/v3-architecture.md)）
- ✅ 简化 `code_embed = Embedding(256, 768)` 实现（B-02 已确定）
- ✅ 与 §3.5 DiscoLoop 表征探针中"Φ 通道"的 vocab 概念对齐（语义上：256 个语义锚点 vs LM 词汇 6400 个）

### 2.2 备选方案（暂不考虑）

| 备选 | 说明 | 否决原因 |
|------|------|----------|
| 改为 `codebook_size=4096` | 匹配 levels=[8,8,8,8] | 64M 强约束下码本利用率无法验证；code_embed 参数量 4×（4096 vs 256×768 = 3.1M vs 0.79M）|
| 改为 `levels=[16,16]` | 256 个组合 | 与现有 FSQ 实现不对齐；需重写量化逻辑 |
| 动态码本 | 按需扩展码字 | 64M 强约束禁止复杂机制 |

---

## 3. SOP 步骤（必须依次执行）

### 3.1 Phase 0 前：文档统一

| # | 任务 | 涉及文档 | 责任方 | 周期 |
|---|------|---------|:---:|:---:|
| 1 | 删除 `levels=[8,8,8,8]` 全部残留 | validation-plan §4.2, v3-arch §3 | SADKO owner | 0.5 天 |
| 2 | 统一标注 `levels=[8,8,4]` 为唯一 64M 配置 | validation-plan §4.2, v3-arch §3, hippo-phase0-manual | SADKO owner | 0.5 天 |
| 3 | 在 v3-arch §3 标注 "禁止 [8,8,8,8]" 防回退 | v3-arch §3 | SADKO owner | 0.5 天 |
| 4 | 更新 open-issues.md B-05 标记已裁决 | open-issues.md §二 B-05, §七 | SADKO owner | 0.5 天 |

**门禁**：4 项全部完成前，禁止启动 Phase 1 v1.0 任何 FSQ 相关实验。

### 3.2 Phase 1 v3.0 中：实施验证

| # | 任务 | 涉及模块 | 周期 |
|---|------|---------|:---:|
| 5 | 实施 `FSQCodeEmbedding(256, 768)` 模块（v3-arch §4.3） | `model/code_embed.py` | 1 天 |
| 6 | FSQ 量化层配置 `levels=[8,8,4]`，输出码字 ∈ [0, 256) | `model/fsq_quantizer.py` | 0.5 天 |
| 7 | 实验 A 通过阈值：码本使用率 > 80%（[v3-arch §6.1](../sadko/v3-architecture.md)） | 64M 实验 A | 2 周 |

**门禁**：实验 A 码本使用率 < 80% → 标记"FSQ 假设失败"，转入 SADKO 负结果清单。

### 3.3 Phase 2 前：下游影响处理

| # | 任务 | 涉及文档 | 责任方 | 周期 |
|---|------|---------|:---:|:---:|
| 8 | 更新 Hippo README 引用此 SOP（标注 codebook_size=256 锁定）| hippo/README.md §2.2 I2 | Hippo owner | 0.5 天 |
| 9 | 更新 four-lines-synthesis §3.4 V4 议题 6 (Meta-Key) 关联此 SOP | four-lines-synthesis.md | Logos/Hippo 协调 | 0.5 天 |
| 10 | 评估 Logos → Hippo FSQ 消费条件（详见 [fsq-consumption-design.md](./fsq-consumption-design.md)） | fsq-consumption-design.md §3 | Logos owner | 1 周 |

---

## 4. 决策矩阵（备查）

| 备选维度 | [8,8,4]=256 ✅ | [8,8,8,8]=4096（clamp）| codebook=4096 |
|---------|:---:|:---:|:---:|
| 码字利用率 | 100% | 6% | 100% |
| code_embed 参数 | 0.79M | 0.79M（被浪费）| 3.1M |
| 64M 训练可行性 | ✅ 已验证 | ✅ 但利用率低 | ❌ 64M 约束无法验证 |
| 300M 升级路径 | 需重训练 | 需重训练（浪费 6%）| 直接 |
| 与 v3-arch §6 实验 A 通过线 | 兼容 | 不通过 | 不适用 |
| 与 DiscoLoop Φ vocab 概念对齐 | ✅ 语义锚点 | ⚠️ 与 LM 词汇混淆 | ⚠️ |
| **推荐度** | **🟢 采用** | 🔴 否决 | 🟡 推迟到 300M |

---

## 5. 风险与回退

| 风险 | 概率 | 影响 | 回退方案 |
|------|:---:|:---:|---------|
| [8,8,4] 量化粒度不足 | 🟡 中 | 中 | 启动 256 → 4096 重训练（300M 阶段） |
| 与 SADKO 历史实现不兼容 | 🟡 中 | 低 | 仅需修改 levels 配置（无架构变更） |
| Logos 消费 FSQ 时 256 不够 | 🟢 低 | 中 | 同上，300M 升格 |
| FSQ 与 DiscoLoop LM-vocab Φ 通道不等价 | 🟠 中 | 中 | 单独实验验证（详见 [fsq-consumption-design.md §4](./fsq-consumption-design.md)） |

---

## 6. 与 Logos 消费 Hippo FSQ 的关系

**关键**：本 SOP 是 [fsq-consumption-design.md](./fsq-consumption-design.md) 的**前置依赖之一**。

只有完成本 SOP 的步骤 1-7（Phase 0 + Phase 1 v3.0 部分）后，Logos 消费 FSQ 的条件才达成：
- ✅ Hippo 配置冲突解决（本 SOP §3.1）
- ⏸️ Hippo 单独机制验证（Phase 1 v3.0 实验 A 通过）
- ⏸️ 跨线对齐（fsq-consumption-design.md §3）

**未完成本 SOP 前，Logos 不应启动 FSQ 消费的任何实验**。

---

## 7. 协调记录

| 时点 | 动作 | 责任人 |
|------|------|--------|
| 2026-07-31 | 创建本 SOP，标记 B-05 待裁决 | DiscoLoop 调研工作流 |
| ⏳ | SADKO owner 审阅 + 裁决 | 待指派 |
| ⏳ | 应用到 SADKO 三份文档 + open-issues.md | 待指派 |
| ⏳ | 启动 Phase 1 v3.0 FSQ 实验 | 待 Phase 0 通过后 |

---

## 8. 相关文档

| 文档 | 关系 |
|------|------|
| [open-issues.md §二 B-05](../sadko/open-issues.md) | 冲突原始描述 |
| [validation-plan §4.2](../sadko/64m-validation-plan.md) | SADKO 64M 计划（待修改）|
| [v3-arch §3](../sadko/v3-architecture.md) | FSQ 详细实现（待修改）|
| [hippo/README.md §2.2 I2](../hippo/README.md) | 胼胝体契约码字协议（待引用）|
| [fsq-consumption-design.md](./fsq-consumption-design.md) | Logos 消费 FSQ 条件设计（前置依赖）|
| [discoloop.md §6.3](../../references/discoloop.md) | DiscoLoop 调研笔记（与 FSQ 等价性论证）|

---

**最后更新**：2026-07-31
**作者**：来自 DiscoLoop 调研工作流（Oracle 评审 → 解决 B-05 → 为 Logos-FSQ 集成铺路）