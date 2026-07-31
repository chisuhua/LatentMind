# Hippo 独立研究线（Hippo Research Line，原名 ELF）

> **定位**：LatentMind 项目第三研究线，专门研究 SADKO 右脑（Hippo）的关键技术——记忆内容 / KG 压缩生长 / FM 检索提取，独立于 SADKO 64M 验证计划推进，后期通过"胼胝体接口契约"接回 SADKO 主干。
> **状态**：🆕 2026-07-30 启动
> **上游 spec**：[docs/superpowers/specs/2026-07-30-hippo-research-line-design.md](../../superpowers/specs/2026-07-30-hippo-research-line-design.md)

---

## 0. 定位声明

- **架构归属**：Hippo 仍是 SADKO 右脑机制（双向注意力 + Flow Matching + FSQ），架构不变
- **解耦维度**：验证路径独立——不复用 SADKO 64M Phase 0-8 流水线，三个子方向各自定义最小验证实验
- **合并路径**：通过本文档 §2 "胼胝体接口契约"接回 SADKO 主干
- **不复用**：不参考 SADKO 64M 验证计划 / Phase 0-8 流水线的时间表（自定节奏）

---

## 1. 三个并行研究方向

| 方向 | 文件 | 核心问题 |
|------|------|---------|
| **方向1：记忆内容架构** | [memory-architecture.md](./memory-architecture.md) | Hippo 内部"知识"的最小表示单元是什么？如何支持增量更新不破坏旧知识？ |
| **方向2：KG 压缩生长** | [graph-growth.md](./graph-growth.md) | KG 结构如何从训练数据自然涌现？新节点/边如何"生长"？ |
| **方向3：FM 检索提取** | [retrieval-extraction.md](./retrieval-extraction.md) | Flow Matching 的 ODE 可逆性如何用于检索？Top-K 精度与速度权衡？ |

**并行性**：三方向最小验证单元可完全独立——见 [上游 spec §3.5](../../superpowers/specs/2026-07-30-hippo-research-line-design.md#35-三个方向的依赖关系与并行性)

---

## 2. 胼胝体接口契约（Corpus Callosum Contract）

> **核心定义**：胼胝体是 Hippo 独立研究线与 SADKO 主干之间的**标准化信息交换接口**。当前为**骨架级契约**（定义接口类别与约束），不锁定具体数值——具体值随 Hippo 研究进展更新。

### 2.1 设计原则

1. **最小化**：只规定 Hippo 与 SADKO 必要的解耦面，不约束 Hippo 内部研究空间
2. **稳定性**：Hippo 内部机制演进不应频繁破坏接口（接口抽象层 vs 实现细节分层）
3. **可验证**：每个 Hippo 研究成果都能映射到接口契约中的某个槽位
4. **对齐基线**：与 SADKO 白皮书 §2.3（Diffusion Alignment）+ §2.4 决策矩阵兼容

### 2.2 接口契约五元组（骨架）

| 接口维度 | 类型 | 输入 | 输出 | 演进约束 |
|---------|------|------|------|---------|
| **I1. Memory KV** | 张量接口 | Hippo 输出的全局语义表示 | `(batch, n_heads, seq_len, head_dim)` 形状的 KV 张量，可被 AR Cross-Attention 读取 | 维度与 AR 的 head_dim 对齐；n_heads 不强求一致 |
| **I2. Codebook Protocol** | 离散码字协议 | 连续向量 | FSQ 索引序列 `idx ∈ ℕ^L`（如 L=3 表示 [8,8,4] 三级码字）| **冻结码字 ↔ 嵌入映射**对左脑只读；码字数量可扩展（如 256 → 1024 → 4096）|
| **I3. Retrieval API** | 异步调用接口 | 查询向量 q（来自左脑或外部）| Top-K 相关码字 + 对应 KV；含置信度分数 | Top-K 默认 K=8；延迟约束：端侧 < 5ms（待 SADKO 主干量化后定）|
| **I4. Incremental Update** | 训练流程接口 | 新知识 KV 输入 | 更新后的码本 + 索引 | 触发条件：新码字招募 / 构象异构 / 模块扩展（见 [sadko/hippo-lifecycle.md](../sadko/hippo-lifecycle.md)）|
| **I5. Failure Fallback** | 降级协议 | 检索失败 / 码本饱和 | 退化到默认值（具体值由实现决定，示例：空 KV + 全 0 嵌入）| 必须支持静默降级，不抛异常 |

### 2.3 关键不变量

无论 Hippo 内部如何演进，以下不变量必须保持：

- **INV-1** Memory KV 输出张量形状满足 I1 约束（左脑可读取）
- **INV-2** Codebook 协议中 `idx → embedding` 映射对左脑**只读冻结**（防止双向修改冲突）
- **INV-3** Retrieval API 返回值必含置信度（左脑可决定是否信任）
- **INV-4** Failure Fallback 路径必须存在且默认行为是"安全降级"
- **INV-5** 接口契约的**变更需双侧 review**（Hippo 研究线 owner + SADKO 主干 owner 共同签字）

### 2.4 演进路径

| 阶段 | 契约状态 | Hippo 研究线状态 | 合并方式 |
|------|---------|--------------|---------|
| **T0（当前）** | 骨架契约（§2.2 + §2.3）| 独立研究，三个方向并行验证 | 尚未合并 |
| **T+1** | 数值化（维度/形状/范围）| Hippo 内部机制初步验证 | 仅 soft 引用，不做硬合并 |
| **T+2** | 接口实现（reference code）| Hippo 三个方向产出 reference implementation | SADKO 64M 可选集成（实验性）|
| **T+3** | 接口锁定（v1.0 contract）| Hippo 三大方向都通过 64M 验证 | SADKO 64M 强制集成（胼胝体启用）|
| **T+4** | 接口演进（v1.x → v2.0）| Hippo 进入规模化（300M+）| SADKO 300M 全量集成 |

### 2.5 与 SADKO 白皮书的兼容性

| SADKO 已有定义 | 胼胝体契约对应 | 一致性 |
|---------------|--------------|--------|
| §2.3 扩散桥梁（Teacher-Student 蒸馏）| I1 Memory KV + I4 Incremental Update | ✅ 一致（蒸馏是 I4 的训练时实现）|
| §2.3 推理时 Zero-Overhead | INV-2（码字映射冻结）| ✅ 一致（推理时不重新计算码本）|
| §2.1 FSQ 几何锚点 | I2 Codebook Protocol | ✅ 一致（FSQ 是码字协议的实现）|
| §7 "右脑 = 海马体" 隐喻 | I3 Retrieval API 语义 | ✅ 一致（海马体 = 检索 + 巩固）|

---

## 3. 验证哲学（继承 SADKO §四）

### 3.1 核心立场

> **Hippo 独立研究线的目标不是产出"最佳性能"，而是产出"鲁棒性评分"与"负结果清单"**。
> 每个机制的"证伪"与"验证"具有同等战略价值。

### 3.2 五大原则

| # | 原则 | Hippo 研究线应用 |
|---|------|--------------|
| **P1** | **消融隔离** | 每个子方向的实验**只改变一个机制变量**，其他保持固定 |
| **P2** | **强制早停** | 64M 阶段任一指标在 30% 训练进度无改善 → 立即停止，标记"未达阈值" |
| **P3** | **鲁棒性 > 性能** | 同一机制在 5+ 随机种子下"全部通过"才算鲁棒；单一种子通过不计入 |
| **P4** | **负结果资产化** | 每个失败实验产出《证伪报告》：失败原因 + 是容量问题还是架构问题 |
| **P5** | **接口契约层验证** | 除机制本身外，必须验证 §2 胼胝体接口契约的五元组可实现 |

### 3.3 鲁棒性评分体系（0-10）

```
鲁棒性评分 =
    (机制有效性 × 0.4)          // 主指标达标率
  + (超参敏感性 × 0.2)          // 1.0 = 不敏感，0.0 = 极度敏感
  + (跨数据集一致性 × 0.2)      // 1.0 = 全数据集通过，0.0 = 单一数据集通过
  + (接口契约满足度 × 0.1)       // §2 五元组是否完整可实现
  + (负结果清晰度 × 0.1)         // 失败案例是否有清晰诊断
```

**判定阈值**：
- 评分 ≥ 8.0：可进入 SADKO 主干集成（T+3 阶段）
- 评分 6.0-7.9：需 300M 中等规模二次验证
- 评分 4.0-5.9：需优化超参 + 多种子验证
- 评分 < 4.0：标记"已证伪"，转入《负结果清单》

---

## 4. 与 `sadko/hippo-*.md` 文档的关系

本目录下的研究是 SADKO-Hippo 既有研究的**独立延伸**（自 2026-07-30 起），不复用 SADKO 64M 验证计划。

| 已有 SADKO-Hippo 文档 | 位置 | 在本目录中的引用方式 |
|------------------|------|---------------------|
| [hippo-vs-gdm-review.md](../sadko/hippo-vs-gdm-review.md) | `sadko/` | **前置依据**：FM 路线裁决（已选定 FM）|
| [hippo-graph-emergence.md](../sadko/hippo-graph-emergence.md) | `sadko/` | **前置依据**：图结构涌现的哲学论证 |
| [hippo-phase0-manual.md](../sadko/hippo-phase0-manual.md) | `sadko/` | **前置依据**：FM+FSQ 基础实验手册 |
| [hippo-lifecycle.md](../sadko/hippo-lifecycle.md) | `sadko/` | **前置依据**：增量更新三模式定义 |

**原则**：已有文档是 SADKO 时期固化研究成果，作为历史技术依据保留原位；本目录下的研究是 SADKO-Hippo 的**未来独立演进**。

---

## 5. 文件清单与状态

| 文件 | 状态 | 议题 | Owner |
|------|------|------|-------|
| [README.md](./README.md) | ✅ 已建立 | 索引 + 胼胝体契约 + 哲学声明 | 待指派 |
| [memory-architecture.md](./memory-architecture.md) | 🆕 骨架 | 方向1：知识存储架构 | 待指派 |
| [graph-growth.md](./graph-growth.md) | 🆕 骨架 | 方向2：KG 压缩生长 | 待指派 |
| [retrieval-extraction.md](./retrieval-extraction.md) | 🆕 骨架 | 方向3：FM 检索提取 | 待指派 |
| [negative-results.md](./negative-results.md) | ⏳ 待建 | 《负结果清单》| 待指派 |

---

## 6. 维护规范

1. **新增研究记录**：单主题一个文件，并在本 README §5 登记
2. **接口契约变更**：必须遵守 §2.3 INV-5（双侧 review）
3. **失败实验**：必须登记到 `negative-results.md`，附诊断（容量/架构/工程化/泛化）
4. **跨文档引用**：使用相对路径（`./`、`../sadko/`、`../../superpowers/specs/`）
5. **语言**：中文为主，英文技术术语保留

---

**下一步**：调用 writing-plans skill 为三个子方向（memory / graph / retrieval）分别生成实施计划。
