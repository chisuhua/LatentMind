# Hippo 独立研究线 — 设计文档

> **状态**：✅ 已确认（待实施）
> **日期**：2026-07-30
> **作者**：Sisyphus (brainstorming skill 产出)
> **目的**：在 LatentMind 项目内建立 Hippo（右脑关键技术）独立研究线，与 SADKO 主干研究并行推进，后期通过胼胝体接口契约合并

---

## 0. 背景与动机

### 0.1 问题陈述

LatentMind 项目当前已建立两条研究线，本设计将引入第三条：

| 研究线 | 定位 | 状态 |
|--------|------|------|
| **Logos 主线** | 推理 + 决策 | ✅ 已有完整文档体系（logos-*）|
| **SADKO 探索分支** | 感知 + 记忆 + 知识 + 多模态 | ✅ 已有完整文档体系（sadko-*），含 Hippo 右脑机制 |
| **Hippo 独立研究线**（本设计引入）| 右脑关键技术独立验证（记忆 / KG / 检索）| 🆕 本设计建立 |

Hippo（海马体，hippocampus 缩写）是 SADKO 右脑的机制名（双向注意力 + Flow Matching + FSQ），已有 4 份专门文档：
- `sadko/hippo-vs-gdm-review.md`（FM vs 图扩散六维裁决）
- `sadko/hippo-graph-emergence.md`（图结构涌现机制）
- `sadko/hippo-phase0-manual.md`（Phase 0 执行手册）
- `sadko/hippo-lifecycle.md`（生命周期管理）

**核心需求**：右脑的三个关键技术——**记忆内容架构 / KG 压缩生长 / FM 检索提取**——需要**独立于 SADKO 64M 验证计划**进行专项研究，避免被现有里程碑绑住节奏。后期通过标准接口（胼胝体契约）合并回 SADKO 主干。

### 0.2 设计目标

1. **组织上独立**：Hippo 研究线作为顶级研究线，与 SADKO / Logos 并列
2. **验证上独立**：不复用 SADKO 64M Phase 0-8 流水线，三个子方向各自定义最小验证实验
3. **接口上稳定**：通过胼胝体契约保证 Hippo 内部演进不破坏 SADKO 集成路径
4. **可并行推进**：三个子方向可同步启动，各自从最小验证单元开始

---

## 1. 目录结构（设计 §1）

### 1.1 最终目录布局

```
docs/research/
│
├── README.md                      (🆕 重写：顶级三研究线索引)
│
├── sadko/                         (🆕 子目录：SADKO 主干研究)
│   ├── README.md
│   ├── whitepaper.md
│   ├── 64m-validation-plan.md
│   ├── v1-architecture.md
│   ├── v2-architecture.md
│   ├── v3-architecture.md
│   ├── multimodal-native.md
│   ├── open-issues.md
│   ├── hippo-vs-gdm-review.md
│   ├── hippo-graph-emergence.md
│   ├── hippo-phase0-manual.md
│   └── hippo-lifecycle.md
│
├── logos/                         (🆕 子目录：Logos 主线研究)
│   ├── README.md
│   ├── whitepaper.md
│   ├── k-strategy.md
│   ├── roadmap.md
│   ├── 64m-validation-plan.md
│   ├── v1-architecture.md
│   ├── v2-architecture.md
│   ├── v3-architecture.md
│   └── sadko-64m-coordination.md
│
└── hippo/                            (🆕 子目录：Hippo 独立研究线 - 本次设计核心)
    ├── README.md                   (索引 + 胼胝体接口契约 + 哲学声明)
    ├── memory-architecture.md      (方向1：知识存储架构)
    ├── graph-growth.md             (方向2：KG 压缩生长)
    └── retrieval-extraction.md     (方向3：FM 检索提取)
```

### 1.2 迁移清单（共 19 份现有文件）

| 现有路径 | 新路径 | 重命名规则 |
|---------|--------|----------|
| `research/sadko-whitepaper.md` | `research/sadko/whitepaper.md` | 去掉 `sadko-` 前缀 |
| `research/sadko-64m-validation-plan.md` | `research/sadko/64m-validation-plan.md` | 同上 |
| `research/sadko-v1-architecture.md` | `research/sadko/v1-architecture.md` | 同上 |
| `research/sadko-v2-architecture.md` | `research/sadko/v2-architecture.md` | 同上 |
| `research/sadko-v3-architecture.md` | `research/sadko/v3-architecture.md` | 同上 |
| `research/sadko-multimodal-native.md` | `research/sadko/multimodal-native.md` | 同上 |
| `research/sadko-open-issues.md` | `research/sadko/open-issues.md` | 同上 |
| `research/sadko-hippo-vs-gdm-review.md` | `research/sadko/hippo-vs-gdm-review.md` | 去掉 `sadko-` 前缀 |
| `research/sadko-hippo-graph-emergence.md` | `research/sadko/hippo-graph-emergence.md` | 同上 |
| `research/sadko-hippo-phase0-manual.md` | `research/sadko/hippo-phase0-manual.md` | 同上 |
| `research/sadko-hippo-lifecycle.md` | `research/sadko/hippo-lifecycle.md` | 同上 |
| `research/logos-whitepaper.md` | `research/logos/whitepaper.md` | 去掉 `logos-` 前缀 |
| `research/logos-k-strategy.md` | `research/logos/k-strategy.md` | 同上 |
| `research/logos-roadmap.md` | `research/logos/roadmap.md` | 同上 |
| `research/logos-64m-validation-plan.md` | `research/logos/64m-validation-plan.md` | 同上 |
| `research/logos-v1-architecture.md` | `research/logos/v1-architecture.md` | 同上 |
| `research/logos-v2-architecture.md` | `research/logos/v2-architecture.md` | 同上 |
| `research/logos-v3-architecture.md` | `research/logos/v3-architecture.md` | 同上 |
| `research/logos-sadko-64m-coordination.md` | `research/logos/sadko-64m-coordination.md` | 仅去 logos- 前缀（保留 sadko- 因它协调两边）|

**重命名原则**：在新子目录下，`sadko-` 和 `logos-` 前缀冗余（已在子目录名中），统一去掉。例外：`logos/sadko-64m-coordination.md` 保留 `sadko-` 前缀强调它是 Logos 协调 SADKO 的文档。

### 1.3 跨文档引用更新

需更新所有引用 `docs/research/` 下文件的路径。重点位置：

| 文件类型 | 引用类型 |
|---------|---------|
| `AGENTS.md` | 硬编码路径（多处）|
| `docs/research/sadko/whitepaper.md` 内部 | 相对路径引用（多处，§三、§六）|
| `docs/research/sadko/multimodal-native.md` | 相对路径（§0、§5、§6、§8）|
| `docs/research/sadko/hippo-*.md` | 相对路径（多处）|
| `docs/research/sadko/v1-v2-v3-architecture.md` | 相对路径（多处）|
| `docs/research/sadko/64m-validation-plan.md` | 相对路径（多处）|
| `docs/research/logos/*.md` | 相对路径（多处）|
| `docs/research/logos/sadko-64m-coordination.md` | 引用 `../implementation/phase-0-*.md` |
| `docs/implementation/README.md` | 待 grep 确认 |
| `docs/architecture.md` | 待 grep 确认 |

**执行策略**：
1. `grep -rn "docs/research/" docs/ AGENTS.md` 找出所有引用
2. 优先使用 `git mv` + `sed` 批量改路径
3. 人工 review 关键交叉引用（SADKO 白皮书内部链接）
4. 在 git commit 中明确标注"docs/research/ 目录重构"以便回溯

### 1.4 顶级 `docs/research/README.md` 重写

新 README 结构：
- §0 三研究线战略定位（Logos + SADKO + Hippo）
- §1 目录组织（三个子目录）
- §2 文件清单（按子目录分类）
- §3 各研究线内部索引（指向 sadko/README.md、logos/README.md、hippo/README.md）
- §4 跨研究线协调文档位置说明
- §5 维护规范（继承原 §4 内容）

---

## 2. 胼胝体接口契约（设计 §2）

### 2.1 核心定义

> **胼胝体（Corpus Callosum）**：Hippo 独立研究线与 SADKO 主干之间的**标准化信息交换接口**。当前为**骨架级契约**（定义接口类别与约束），不锁定具体数值——具体值随 Hippo 研究进展更新。

### 2.2 设计原则

1. **最小化**：只规定 Hippo 与 SADKO 必要的解耦面，不约束 Hippo 内部研究空间
2. **稳定性**：Hippo 内部机制演进不应频繁破坏接口（接口抽象层 vs 实现细节分层）
3. **可验证**：每个 Hippo 研究成果都能映射到接口契约中的某个槽位
4. **对齐基线**：与 SADKO 白皮书 §2.3（Diffusion Alignment）+ §2.4 决策矩阵兼容

### 2.3 接口契约五元组（骨架）

| 接口维度 | 类型 | 输入 | 输出 | 演进约束 |
|---------|------|------|------|---------|
| **I1. Memory KV** | 张量接口 | Hippo 输出的全局语义表示 | `(batch, n_heads, seq_len, head_dim)` 形状的 KV 张量，可被 AR Cross-Attention 读取 | 维度与 AR 的 head_dim 对齐；n_heads 不强求一致 |
| **I2. Codebook Protocol** | 离散码字协议 | 连续向量 | FSQ 索引序列 `idx ∈ ℕ^L`（如 L=3 表示 [8,8,4] 三级码字）| **冻结码字 ↔ 嵌入映射**对左脑只读；码字数量可扩展（如 256 → 1024 → 4096）|
| **I3. Retrieval API** | 异步调用接口 | 查询向量 q（来自左脑或外部）| Top-K 相关码字 + 对应 KV；含置信度分数 | Top-K 默认 K=8；延迟约束：端侧 < 5ms（待 SADKO 主干量化后定）|
| **I4. Incremental Update** | 训练流程接口 | 新知识 KV 输入 | 更新后的码本 + 索引 | 触发条件：新码字招募 / 构象异构 / 模块扩展（见 sadko/hippo-lifecycle.md）|
| **I5. Failure Fallback** | 降级协议 | 检索失败 / 码本饱和 | 退化到默认值（具体值由实现决定，示例：空 KV + 全 0 嵌入）| 必须支持静默降级，不抛异常 |

### 2.4 接口契约的关键不变量

无论 Hippo 内部如何演进，以下不变量必须保持：

- **INV-1** Memory KV 输出张量形状满足 I1 约束（左脑可读取）
- **INV-2** Codebook 协议中 `idx → embedding` 映射对左脑**只读冻结**（防止双向修改冲突）
- **INV-3** Retrieval API 返回值必含置信度（左脑可决定是否信任）
- **INV-4** Failure Fallback 路径必须存在且默认行为是"安全降级"
- **INV-5** 接口契约的**变更需双侧 review**（Hippo 研究线 owner + SADKO 主干 owner 共同签字）

### 2.5 演进路径

| 阶段 | 契约状态 | Hippo 研究线状态 | 合并方式 |
|------|---------|--------------|---------|
| **当前（T0）** | 骨架契约（§2.3 + §2.4）| 独立研究，三个方向并行验证 | 尚未合并 |
| **T+1** | 数值化（维度/形状/范围）| Hippo 内部机制初步验证 | 仅 soft 引用，不做硬合并 |
| **T+2** | 接口实现（reference code）| Hippo 三个方向产出 reference implementation | SADKO 64M 可选集成（实验性）|
| **T+3** | 接口锁定（v1.0 contract）| Hippo 三大方向都通过 64M 验证 | SADKO 64M 强制集成（胼胝体启用）|
| **T+4** | 接口演进（v1.x → v2.0）| Hippo 进入规模化（300M+）| SADKO 300M 全量集成 |

### 2.6 与 SADKO 白皮书的兼容性

| SADKO 已有定义 | 胼胝体契约对应 | 一致性 |
|---------------|--------------|--------|
| §2.3 扩散桥梁（Teacher-Student 蒸馏）| I1 Memory KV + I4 Incremental Update | ✅ 一致（蒸馏是 I4 的训练时实现）|
| §2.3 推理时 Zero-Overhead | INV-2（码字映射冻结）| ✅ 一致（推理时不重新计算码本）|
| §2.1 FSQ 几何锚点 | I2 Codebook Protocol | ✅ 一致（FSQ 是码字协议的实现）|
| §7 "右脑 = 海马体" 隐喻 | I3 Retrieval API 语义 | ✅ 一致（海马体 = 检索 + 巩固）|

---

## 3. 三个子方向的研究边界（设计 §3）

### 3.1 子方向总览

```
┌─────────────────────────────────────────────────────────────────┐
│                Hippo 独立研究线（hippo/ 目录）                       │
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────┐  │
│  │ 方向1：         │    │ 方向2：         │    │ 方向3：         │  │
│  │ 记忆内容架构    │    │ KG 压缩生长     │    │ FM 检索提取     │  │
│  │ (memory-arch)  │    │ (graph-growth) │    │ (retrieval)    │  │
│  │               │    │               │    │                │  │
│  │ "知识怎么存"    │ →  │ "知识怎么长成图" │ →  │ "知识怎么取"    │  │
│  └───────┬───────┘    └───────┬───────┘    └────────┬───────┘  │
│          │                    │                     │          │
│          └────────────────────┴─────────────────────┘          │
│                              ↓                                  │
│                    胼胝体接口契约（§2）                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 方向 1：记忆内容架构（memory-architecture.md）

| 项 | 内容 |
|----|------|
| **核心问题** | Hippo 内部"知识"的最小表示单元是什么？知识如何在压缩空间中保持关联？增量更新如何不破坏已有知识？ |
| **研究目标** | 定义 Hippo 知识存储的最优架构：压缩率、重构质量、增量稳定性三者的 Pareto 最优 |
| **最小验证单元** | `MemoryStore` 模块：(a) 接受 KV 输入 (b) 输出压缩码字 + 关联图 (c) 支持增量插入不破坏旧知识 |
| **关键指标** | (a) 压缩比（输入 KV tokens / 输出码字）(b) 重构 MSE (c) 增量 N 次后旧知识保留率 |
| **前置依赖** | `sadko/hippo-vs-gdm-review.md`（FM 路线已选）、`sadko/hippo-lifecycle.md`（增量模式定义）|
| **输出物** | `hippo/memory-architecture.md`：知识表示 schema、压缩算法、增量更新算法、参考实现 |

### 3.3 方向 2：KG 压缩生长（graph-growth.md）

| 项 | 内容 |
|----|------|
| **核心问题** | KG 结构如何从训练数据自然涌现？码字间关联如何形成图边？新节点/边如何"生长"？生长与压缩饱和的关系？ |
| **研究目标** | 理解 KG 的"涌现 → 生长 → 饱和"全生命周期，找到促生长训练策略的最优组合 |
| **最小验证单元** | 小规模 KG（100-1000 节点）训练 Hippo，观察图结构自发涌现：(a) 聚类系数 (b) 平均路径长度 (c) 度分布 |
| **关键指标** | (a) 涌现速度（多少步后图论指标稳定）(b) 节点度分布是否符合 power-law (c) 新节点引入的全局重组织幅度 |
| **前置依赖** | `sadko/hippo-graph-emergence.md`（涌现的哲学论证）；可选 `memory-architecture` 方向的部分输出 |
| **输出物** | `hippo/graph-growth.md`：促生长训练策略（数据几何压力 / 拓扑软约束 / 架构引导）、图论验证工具箱、生长曲线 |

### 3.4 方向 3：FM 检索提取（retrieval-extraction.md）

| 项 | 内容 |
|----|------|
| **核心问题** | FM 的 ODE 可逆性如何用于检索？ODE 方向如何选择？Top-K 精度与速度权衡？碎片化关联如何提取？ |
| **研究目标** | 基于 Flow Matching 的检索模块：Recall@K ≥ 0.9、端侧 < 5ms、支持模糊跨域联想 |
| **最小验证单元** | `FMRetrieval` 模块：(a) 输入 query 向量 (b) 输出 Top-K 码字 + 关联 KV (c) 支持 Slerp 模糊联想 |
| **关键指标** | (a) Recall@K (b) 检索延迟 (c) 跨码字联想质量（人工评估 + 自动化指标）(d) 失败降级率 |
| **前置依赖** | `sadko/hippo-phase0-manual.md`（FM+FSQ 基础）；可选 `memory-architecture` 的码字 schema |
| **输出物** | `hippo/retrieval-extraction.md`：FM 检索算法、ODE 方向选择策略、Slerp 联想测试、失败降级协议 |

### 3.5 三个方向的依赖关系与并行性

```
memory-architecture  ←─── graph-growth   ←─── retrieval-extraction
     (基础)              (依赖部分 memory)     (依赖 memory + graph)

但每个方向的最小验证单元可以**完全独立**：
- memory: 在空 KG 上做 KV → 码字压缩
- graph: 在人造小 KG 上观察涌现（无需真实码字）
- retrieval: 在人造码字集上做检索（无需真实 KG）
```

**并行推进策略**：
1. 三方向同步启动，各自从最小验证单元开始
2. 每个方向产出 reference implementation 后，考虑跨方向集成（软合并）
3. 三方向全部完成 64M 验证后，进入 SADKO 主干集成（T+3 阶段）

### 3.6 子方向间的接口约定（防止内部耦合）

| 接口 | 提供方 | 使用方 | 约束 |
|------|--------|--------|------|
| **码字 schema**（码字的表示形式）| memory | graph / retrieval | 码字 = L 维整数索引（每个分量是 0..Nᵢ-1 的整数）；L 默认 3（可扩展，对应 FSQ 多级量化）|
| **码字 ↔ 嵌入映射** | memory | graph / retrieval | 只读冻结；首次发布后不修改 |
| **图查询 API** | graph | retrieval | 输入：源码字 + 关系类型；输出：关联码字集 |
| **检索 query 类型** | retrieval | —（被外部调用）| 输入：连续向量或码字 |

**原则**：每个子方向只暴露上述接口中的对应部分，**不直接依赖其他子方向的内部实现**。

---

## 4. 验证哲学与鲁棒性评分（设计 §4）

### 4.1 核心立场

> **Hippo 独立研究线的目标不是产出"最佳性能"，而是产出"鲁棒性评分"与"负结果清单"**。
> 每个机制的"证伪"与"验证"具有同等战略价值。

### 4.2 五大原则

| # | 原则 | Hippo 研究线应用 |
|---|------|--------------|
| **P1** | **消融隔离** | 每个子方向的实验**只改变一个机制变量**，其他保持固定 |
| **P2** | **强制早停** | 64M 阶段任一指标在 30% 训练进度无改善 → 立即停止，标记"未达阈值" |
| **P3** | **鲁棒性 > 性能** | 同一机制在 5+ 随机种子下"全部通过"才算鲁棒；单一种子通过不计入 |
| **P4** | **负结果资产化** | 每个失败实验产出《证伪报告》：失败原因 + 是容量问题还是架构问题 |
| **P5** | **接口契约层验证** | 除机制本身外，必须验证 §2 胼胝体接口契约的五元组可实现 |

### 4.3 病理学诊断（区分容量 vs 架构缺陷）

| 现象 | 诊断 | 处理 |
|------|------|------|
| Loss 持续下降但未收敛 | **容量问题**（机制有效，需放大）| 升级到 300M 验证 |
| Loss 早期停滞或剧烈震荡 | **架构问题**（机制无效，需废弃）| 标记"已证伪"，转入《证伪清单》|
| 指标在 64M 需精调 5+ 超参才能工作 | **工程化不友好** | 标记"鲁棒性不足"，转 Plan B 备选机制 |
| 同一机制在多数据集下行为不一致 | **泛化失败** | 标记"过拟合数据集"，需多数据集联合验证 |

### 4.4 鲁棒性评分体系（量化标准）

每个子方向产出最终鲁棒性评分（0-10）：

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

### 4.5 验证实验矩阵（每个子方向）

每个子方向必须至少包含：

| 实验类型 | 目的 | 失败判定 |
|---------|------|---------|
| **A. 最小验证** | 验证机制在最小单元下能工作 | 最小单元失败 → 机制废弃 |
| **B. 消融对照** | 隔离机制 vs 基线，确认贡献 | 消融后性能无显著下降 → 机制可能是冗余 |
| **C. 超参扫描** | 5+ 关键超参的敏感性测试 | 任一超参需精细调节 → 鲁棒性不足 |
| **D. 多数据集** | ≥ 3 个数据集验证泛化 | 跨数据集行为不一致 → 过拟合 |
| **E. 接口契约测试** | 验证 §2 五元组的可实现性 | 接口契约不可满足 → 架构不可集成 |

### 4.6 与 SADKO 64M 验证计划的关系

| 维度 | SADKO 64M 验证计划 | Hippo 独立研究线 |
|------|-------------------|---------------|
| **目标** | 端到端模型架构证伪 | 关键技术独立验证 |
| **验证单位** | 整个模型在任务上的表现 | 单个机制在隔离环境下的表现 |
| **时间表** | 10 周（已有计划）| **自定节奏**，不与 SADKO 64M 同步 |
| **资源** | 单卡 3090 共用 | 可与 SADKO 并行使用同一卡（分时）|
| **合并时点** | — | T+3 阶段（见 §2.5）|

**关键区别**：Hippo 研究线**不复用 SADKO 64M 验证计划的 Phase 0-8 流水线**，三个子方向各自定义最小验证实验。

### 4.7 产出物清单

每个子方向完成后必须产出：

- [ ] **机制报告**（每个机制 1 份）：设计原理、实现细节、验证结果
- [ ] **参考实现**（每个机制 1 份代码）：PyTorch 模块 + 单元测试
- [ ] **鲁棒性评分**（每个子方向 1 份）：按 §4.4 评分
- [ ] **《负结果清单》条目**（每个失败实验 1 行）：失败原因 + 诊断 + 是否转入 Plan B

---

## 5. 实施路径

### 5.1 阶段划分

| 阶段 | 时间 | 任务 | 验收 |
|------|------|------|------|
| **Phase A：目录重构** | T0 ~ T0+1d | 全量迁移 19 份现有文档 + 创建 hippo/ 子目录（含 4 份新文档）| 所有跨文档引用工作；git commit 提交 |
| **Phase B：胼胝体契约文档化** | T0+1d ~ T0+3d | 完成 `hippo/README.md`（含 §2 接口契约五元组 + §2.4 不变量）| SADKO 主干 owner review 通过 |
| **Phase C：三个子方向启动** | T0+3d 起 | 并行启动 memory / graph / retrieval 三个方向的最小验证 | 各自最小验证单元通过 |
| **Phase D：参考实现产出** | T+3d 起（与 Phase C 衔接）| 三个方向各自产出 reference implementation + 单元测试 | 通过 §4.5 验证实验矩阵 |

### 5.2 关键依赖

- Phase A 必须先完成（文档位置稳定后，跨引用才能正确解析）
- Phase B 在 Phase A 完成后立即执行（胼胝体契约是后续所有讨论的基础）
- Phase C 可与 Phase B 并行（子方向的最小验证不依赖胼胝体契约的最终版）

### 5.3 风险与降级

| 风险 | 概率 | 影响 | 降级预案 |
|------|:---:|:---:|---------|
| 跨文档引用 grep 漏检导致 broken link | 中 | 中 | 用 `markdown-link-check` 等工具二次扫描 |
| 用户对迁移清单有异议（某些文件位置想调整）| 中 | 低 | 在执行前再次 review 清单 |
| hippo/ 子目录命名与项目未来演进冲突 | 低 | 低 | 子目录名是物理组织，未来重命名成本可控 |
| 三个子方向的 reference implementation 互相依赖导致不能完全并行 | 中 | 中 | 通过 §3.6 子方向间接口约定强制隔离 |

---

## 6. 决策记录

| 时间 | 决策点 | 决策 | 理由 |
|------|--------|------|------|
| 2026-07-30 | 命名方案 | A+B（子目录 + hippo-* 命名）| 物理隔离最强 + 复用 Hippo 术语资产 |
| 2026-07-30 | 解耦边界 | 验证路径解耦，架构不变 | 用户明确"架构没有改变" |
| 2026-07-30 | 目录重构范围 | 全量迁移所有 19 份文档 | 用户选择"全量迁移"以保证组织统一 |
| 2026-07-30 | 胼胝体契约粒度 | 骨架契约（不锁定数值）| Hippo 内部尚未稳定，过早数值化会反复修改契约 |
| 2026-07-30 | 三个子方向依赖 | 各方向最小验证单元可独立 | 满足"并行推进"的核心需求 |
| 2026-07-30 | 验证哲学 | 继承 SADKO §四 + 鲁棒性评分量化 | 与 SADKO 文化一致，但适配独立研究线特点 |

---

## 7. 开放问题（实施时待解决）

1. **子目录中 README.md 的具体内容**：每个子目录的 README 需要把原 docs/research/README.md §3.1/§3.2/§3.3 的对应部分迁移过来——具体哪些内容放 README、哪些放子文件待定
2. **跨文档引用更新是否做 git mv + sed 批量**：需要测试 sed 在中文 markdown 中的稳定性
3. **三个子方向的 owner 分配**：本次未指定——需用户后续指定
4. **胼胝体契约的 version 字段**：v1.0 contract 锁定时（T+3）需要明确的版本号管理策略
5. **与 SADKO 协调中枢（logos/sadko-64m-coordination.md）的双向引用关系**：logos 文档会引用 hippo/ 文档吗？还是 hippo/ 文档完全单向引用 sadko/ 文档？待 SADKO 主干 owner review 时确认

---

**下一步**：调用 writing-plans skill，根据本设计生成实施计划。