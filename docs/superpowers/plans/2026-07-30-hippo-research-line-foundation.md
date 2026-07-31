# Hippo 独立研究线基础设施 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 `docs/research/` 为 sadko/logos/hippo 三子目录结构，迁移 19 份现有文档并创建 `hippo/README.md`（含 §2 胼胝体接口契约 + 五个不变量），为 Hippo 独立研究线建立稳定基础设施。

**Architecture:** Phase A：迁移现有 docs/research/ 文件到三子目录（含路径去前缀 + 跨文档引用更新）。Phase B：创建 hippo/ 子目录骨架（含 README.md 完整胼胝体契约 + 三子方向骨架文件）。Phase C：重写顶级 docs/research/README.md 为三研究线索引。Phase D：更新 AGENTS.md 等外部引用。

**Tech Stack:** git、grep、sed、markdown 验证工具、bash 命令

**上游 spec:** `docs/superpowers/specs/2026-07-30-hippo-research-line-design.md`

---

## 文件结构总览

### 新建文件
- `docs/research/sadko/README.md`
- `docs/research/logos/README.md`
- `docs/research/hippo/README.md`（核心：含胼胝体契约五元组 + 五个不变量）
- `docs/research/hippo/memory-architecture.md`（骨架）
- `docs/research/hippo/graph-growth.md`（骨架）
- `docs/research/hippo/retrieval-extraction.md`（骨架）

### 迁移文件（git mv）
SADKO（11 份）→ `docs/research/sadko/`：
- `whitepaper.md`、`64m-validation-plan.md`、`v1-architecture.md`、`v2-architecture.md`、`v3-architecture.md`、`multimodal-native.md`、`open-issues.md`、`hippo-vs-gdm-review.md`、`hippo-graph-emergence.md`、`hippo-phase0-manual.md`、`hippo-lifecycle.md`

Logos（8 份）→ `docs/research/logos/`：
- `whitepaper.md`、`k-strategy.md`、`roadmap.md`、`64m-validation-plan.md`、`v1-architecture.md`、`v2-architecture.md`、`v3-architecture.md`、`sadko-64m-coordination.md`

### 修改文件
- `docs/research/README.md`（重写为三研究线索引）
- `AGENTS.md`（更新硬编码路径）
- `docs/architecture.md`（如引用了 docs/research/ 路径）
- `docs/implementation/README.md`（如引用了 docs/research/ 路径）

---

### Task 1: 准备工作 — 创建子目录 + 扫描跨文档引用

**Files:**
- Create: `docs/research/sadko/`、`docs/research/logos/`、`docs/research/hippo/`（空目录）
- Create: `docs/research/REFACTOR_MAP.md`（迁移清单 + 路径映射表）

- [ ] **Step 1: 创建三个空子目录**

```bash
mkdir -p /workspace/project/LatentMind/docs/research/sadko
mkdir -p /workspace/project/LatentMind/docs/research/logos
mkdir -p /workspace/project/LatentMind/docs/research/hippo
ls -la /workspace/project/LatentMind/docs/research/
```

Expected output: 看到 `sadko/`、`logos/`、`hippo/` 三个新目录（可能为空）

- [ ] **Step 2: 扫描所有引用 docs/research/ 的文件**

```bash
cd /workspace/project/LatentMind
grep -rn "docs/research/" docs/ AGENTS.md README.md 2>/dev/null | grep -v "^Binary" | sort -u
```

Expected: 输出引用 docs/research/ 下文件的所有位置（包括文件名 + 行号 + 引用内容）

- [ ] **Step 3: 生成引用清单文档**

将上一步输出保存：

```bash
cd /workspace/project/LatentMind
grep -rn "docs/research/" docs/ AGENTS.md README.md 2>/dev/null | grep -v "^Binary" | sort -u > docs/research/REFACTOR_MAP.md
wc -l docs/research/REFACTOR_MAP.md
```

Expected: REFACTOR_MAP.md 包含所有引用位置（可能数十行）

- [ ] **Step 4: 验证准备工作完成**

```bash
cd /workspace/project/LatentMind
[ -d docs/research/sadko ] && [ -d docs/research/logos ] && [ -d docs/research/hippo ] && [ -s docs/research/REFACTOR_MAP.md ] && echo "OK: 三子目录 + REFACTOR_MAP.md 就绪" || echo "FAIL"
```

Expected: `OK: 三子目录 + REFACTOR_MAP.md 就绪`

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/
git commit -m "chore(research): 创建 sadko/logos/hippo 三子目录 + REFACTOR_MAP 引用清单"
```

---

### Task 2: 迁移 SADKO 文档到 sadko/ 子目录（11 份）

**Files:**
- Move (git mv): 11 份 SADKO 文档
- Modify: 内部相对引用（去掉 `sadko-` 前缀）

- [ ] **Step 1: 批量迁移 11 份 SADKO 文档**

```bash
cd /workspace/project/LatentMind
git mv docs/research/sadko-whitepaper.md docs/research/sadko/whitepaper.md
git mv docs/research/sadko-64m-validation-plan.md docs/research/sadko/64m-validation-plan.md
git mv docs/research/sadko-v1-architecture.md docs/research/sadko/v1-architecture.md
git mv docs/research/sadko-v2-architecture.md docs/research/sadko/v2-architecture.md
git mv docs/research/sadko-v3-architecture.md docs/research/sadko/v3-architecture.md
git mv docs/research/sadko-multimodal-native.md docs/research/sadko/multimodal-native.md
git mv docs/research/sadko-open-issues.md docs/research/sadko/open-issues.md
git mv docs/research/sadko-hippo-vs-gdm-review.md docs/research/sadko/hippo-vs-gdm-review.md
git mv docs/research/sadko-hippo-graph-emergence.md docs/research/sadko/hippo-graph-emergence.md
git mv docs/research/sadko-hippo-phase0-manual.md docs/research/sadko/hippo-phase0-manual.md
git mv docs/research/sadko-hippo-lifecycle.md docs/research/sadko/hippo-lifecycle.md
ls docs/research/sadko/
```

Expected: 列出 11 份 .md 文件（无 `sadko-` 前缀）

- [ ] **Step 2: 验证 SADKO 文档已全部移走（根目录应无 sadko-* 文件）**

```bash
cd /workspace/project/LatentMind
ls docs/research/ | grep -E "^sadko-" || echo "OK: 无 sadko-* 文件残留"
```

Expected: `OK: 无 sadko-* 文件残留`

- [ ] **Step 3: 更新 SADKO 文档内部的相对路径引用**

```bash
cd /workspace/project/LatentMind
# 批量更新 SADKO 文档内部引用（如 [sadko-whitepaper.md](./sadko-whitepaper.md) → [whitepaper.md](./whitepaper.md)）
# 注意：sadko-hippo-* → hippo-* 也需要更新
sed -i 's|sadko-whitepaper\.md|whitepaper.md|g; s|sadko-64m-validation-plan\.md|64m-validation-plan.md|g; s|sadko-v1-architecture\.md|v1-architecture.md|g; s|sadko-v2-architecture\.md|v2-architecture.md|g; s|sadko-v3-architecture\.md|v3-architecture.md|g; s|sadko-multimodal-native\.md|multimodal-native.md|g; s|sadko-open-issues\.md|open-issues.md|g; s|sadko-hippo-vs-gdm-review\.md|hippo-vs-gdm-review.md|g; s|sadko-hippo-graph-emergence\.md|hippo-graph-emergence.md|g; s|sadko-hippo-phase0-manual\.md|hippo-phase0-manual.md|g; s|sadko-hippo-lifecycle\.md|hippo-lifecycle.md|g' docs/research/sadko/*.md
echo "SADKO 内部引用更新完成"
```

Expected: 无错误输出，"SADKO 内部引用更新完成"

- [ ] **Step 4: 验证 SADKO 文档内部无旧引用残留**

```bash
cd /workspace/project/LatentMind
# 检查 sadko/ 子目录内是否还有对 sadko-*.md 的引用（注意：保留对 ../sadko/sadko-64m-coordination.md 等合法引用）
grep -rn "sadko-whitepaper\|sadko-64m-validation\|sadko-v[123]-architecture\|sadko-multimodal\|sadko-open-issues\|sadko-hippo-" docs/research/sadko/ || echo "OK: 无 SADKO 旧前缀引用残留"
```

Expected: `OK: 无 SADKO 旧前缀引用残留`

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/sadko/
git status --short
git commit -m "refactor(research): 迁移 11 份 SADKO 文档到 sadko/ 子目录（去前缀）"
```

---

### Task 3: 迁移 Logos 文档到 logos/ 子目录（8 份）

**Files:**
- Move (git mv): 8 份 Logos 文档
- Modify: 内部相对引用（去掉 `logos-` 前缀；保留 `sadko-` 前缀于 sadko-64m-coordination.md）

- [ ] **Step 1: 批量迁移 8 份 Logos 文档**

```bash
cd /workspace/project/LatentMind
git mv docs/research/logos-whitepaper.md docs/research/logos/whitepaper.md
git mv docs/research/logos-k-strategy.md docs/research/logos/k-strategy.md
git mv docs/research/logos-roadmap.md docs/research/logos/roadmap.md
git mv docs/research/logos-64m-validation-plan.md docs/research/logos/64m-validation-plan.md
git mv docs/research/logos-v1-architecture.md docs/research/logos/v1-architecture.md
git mv docs/research/logos-v2-architecture.md docs/research/logos/v2-architecture.md
git mv docs/research/logos-v3-architecture.md docs/research/logos/v3-architecture.md
git mv docs/research/logos-sadko-64m-coordination.md docs/research/logos/sadko-64m-coordination.md
ls docs/research/logos/
```

Expected: 列出 8 份 .md 文件（含 `sadko-64m-coordination.md`）

- [ ] **Step 2: 验证 Logos 文档已全部移走**

```bash
cd /workspace/project/LatentMind
ls docs/research/ | grep -E "^logos-" || echo "OK: 无 logos-* 文件残留"
```

Expected: `OK: 无 logos-* 文件残留`

- [ ] **Step 3: 更新 Logos 文档内部的相对路径引用**

```bash
cd /workspace/project/LatentMind
# 注意：只去掉 logos- 前缀，保留 sadko- 前缀（因为 logos 文档可能引用 sadko-* 文档）
sed -i 's|logos-whitepaper\.md|whitepaper.md|g; s|logos-k-strategy\.md|k-strategy.md|g; s|logos-roadmap\.md|roadmap.md|g; s|logos-64m-validation-plan\.md|64m-validation-plan.md|g; s|logos-v1-architecture\.md|v1-architecture.md|g; s|logos-v2-architecture\.md|v2-architecture.md|g; s|logos-v3-architecture\.md|v3-architecture.md|g' docs/research/logos/*.md
echo "Logos 内部引用更新完成"
```

Expected: 无错误输出，"Logos 内部引用更新完成"

- [ ] **Step 4: 验证 Logos 文档内部无旧引用残留（除 sadko-64m-coordination.md 自身）**

```bash
cd /workspace/project/LatentMind
grep -rn "logos-whitepaper\|logos-k-strategy\|logos-roadmap\|logos-64m-validation\|logos-v[123]-architecture" docs/research/logos/ || echo "OK: 无 Logos 旧前缀引用残留"
```

Expected: `OK: 无 Logos 旧前缀引用残留`

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/logos/
git commit -m "refactor(research): 迁移 8 份 Logos 文档到 logos/ 子目录（去前缀）"
```

---

### Task 4: 创建 hippo/README.md（含胼胝体接口契约）

**Files:**
- Create: `docs/research/hippo/README.md`

- [ ] **Step 1: 写完整的 hippo/README.md**

写入以下内容到 `docs/research/hippo/README.md`（含 §0 定位 + §1 三子方向索引 + §2 胼胝体契约五元组 + §3 五个不变量 + §4 演进路径 + §5 与 sadko/hippo-* 文档的关系）：

```bash
cat > /workspace/project/LatentMind/docs/research/hippo/README.md << 'EOF'
# Hippo 独立研究线（Hippo Research Line）

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
| [README.md](./README.md) | ✅ 已建立 | 索引 + 胼胝体契约 + 哲学声明 | TBD |
| [memory-architecture.md](./memory-architecture.md) | 🆕 骨架 | 方向1：知识存储架构 | TBD |
| [graph-growth.md](./graph-growth.md) | 🆕 骨架 | 方向2：KG 压缩生长 | TBD |
| [retrieval-extraction.md](./retrieval-extraction.md) | 🆕 骨架 | 方向3：FM 检索提取 | TBD |
| [negative-results.md](./negative-results.md) | ⏳ 待建 | 《负结果清单》| TBD |

---

## 6. 维护规范

1. **新增研究记录**：单主题一个文件，并在本 README §5 登记
2. **接口契约变更**：必须遵守 §2.3 INV-5（双侧 review）
3. **失败实验**：必须登记到 `negative-results.md`，附诊断（容量/架构/工程化/泛化）
4. **跨文档引用**：使用相对路径（`./`、`../sadko/`、`../../superpowers/specs/`）
5. **语言**：中文为主，英文技术术语保留

---

**下一步**：调用 writing-plans skill 为三个子方向（memory / graph / retrieval）分别生成实施计划。
EOF
echo "hippo/README.md 已创建"
```

Expected: 无错误输出，"hippo/README.md 已创建"

- [ ] **Step 2: 验证 README.md 格式正确**

```bash
cd /workspace/project/LatentMind
# 检查文件存在
[ -f docs/research/hippo/README.md ] && echo "OK: README.md 存在" || echo "FAIL"
# 检查核心契约章节存在
grep -q "## 2. 胼胝体接口契约" docs/research/hippo/README.md && echo "OK: §2 契约章节存在" || echo "FAIL: §2 契约章节缺失"
grep -q "### 2.2 接口契约五元组" docs/research/hippo/README.md && echo "OK: §2.2 五元组存在" || echo "FAIL: §2.2 五元组缺失"
grep -q "### 2.3 关键不变量" docs/research/hippo/README.md && echo "OK: §2.3 不变量存在" || echo "FAIL: §2.3 不变量缺失"
```

Expected: 4 行 `OK: ...` 输出

- [ ] **Step 3: 验证 Markdown 链接可解析**

```bash
cd /workspace/project/LatentMind
# 检查内部相对引用至少存在（不要求全部解析，避免链接检查器依赖）
grep -E "\.\./sadko/hippo-" docs/research/hippo/README.md | head -5
grep -E "\./memory-architecture\.md" docs/research/hippo/README.md | head -3
```

Expected: 看到至少 5 行 ../sadko/hippo-* 引用 + 3 行 ./memory-architecture.md 引用

- [ ] **Step 4: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/hippo/README.md
git commit -m "feat(hippo): 创建 hippo/README.md 含胼胝体接口契约五元组 + 五个不变量"
```

---

### Task 5: 创建三个 hippo 子方向骨架文件

**Files:**
- Create: `docs/research/hippo/memory-architecture.md`
- Create: `docs/research/hippo/graph-growth.md`
- Create: `docs/research/hippo/retrieval-extraction.md`

- [ ] **Step 1: 创建 memory-architecture.md 骨架**

```bash
cat > /workspace/project/LatentMind/docs/research/hippo/memory-architecture.md << 'EOF'
# Hippo 记忆内容架构（Memory Architecture）

> **方向**：方向 1 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)

---

## 0. 核心问题

Hippo 内部"知识"的最小表示单元是什么？知识如何在压缩空间中保持关联？增量更新如何不破坏已有知识？

---

## 1. 研究目标

定义 Hippo 知识存储的最优架构：压缩率、重构质量、增量稳定性三者的 Pareto 最优。

---

## 2. 最小验证单元

`MemoryStore` 模块：
- (a) 接受 KV 输入
- (b) 输出压缩码字 + 关联图
- (c) 支持增量插入不破坏旧知识

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| 压缩比 | 输入 KV tokens / 输出码字 | ≥ 8:1 |
| 重构 MSE | Flow Matching 解压后的 KV 误差 | < 0.05 |
| 增量稳定性 | 增量 N 次后旧知识保留率 | > 95% |

---

## 4. 前置依赖

- [`sadko/hippo-vs-gdm-review.md`](../sadko/hippo-vs-gdm-review.md)（FM 路线已选）
- [`sadko/hippo-lifecycle.md`](../sadko/hippo-lifecycle.md)（增量模式定义）

---

## 5. 待填充内容（实施时）

- [ ] 知识表示 schema（码字结构、嵌入维度、关联图边类型）
- [ ] 压缩算法（FM 编码器 + FSQ 离散化）
- [ ] 增量更新算法（码字招募 / 构象异构 / 模块扩展）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
EOF
echo "memory-architecture.md 已创建"
```

Expected: 无错误输出

- [ ] **Step 2: 创建 graph-growth.md 骨架**

```bash
cat > /workspace/project/LatentMind/docs/research/hippo/graph-growth.md << 'EOF'
# Hippo 知识图谱压缩生长（KG Compressed Growth）

> **方向**：方向 2 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)

---

## 0. 核心问题

KG 结构如何从训练数据自然涌现？码字间关联如何形成图边？新节点/边如何"生长"？生长与压缩饱和的关系？

---

## 1. 研究目标

理解 KG 的"涌现 → 生长 → 饱和"全生命周期，找到促生长训练策略的最优组合。

---

## 2. 最小验证单元

小规模 KG（100-1000 节点）训练 Hippo，观察图结构自发涌现：
- (a) 聚类系数
- (b) 平均路径长度
- (c) 节点度分布

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| 涌现速度 | 多少步后图论指标稳定 | < 10K 步 |
| 度分布 | 节点度是否符合 power-law | α ∈ [2, 3] |
| 重组织幅度 | 新节点引入的全局结构调整 | < 20% |

---

## 4. 前置依赖

- [`sadko/hippo-graph-emergence.md`](../sadko/hippo-graph-emergence.md)（涌现的哲学论证）
- 可选：[`memory-architecture.md`](./memory-architecture.md) 方向的部分输出

---

## 5. 待填充内容（实施时）

- [ ] 促生长训练策略（数据几何压力 / 拓扑软约束 / 架构引导）
- [ ] 图论验证工具箱（NetworkX / igraph 集成）
- [ ] 生长曲线监控（节点数 vs 时间、聚类系数 vs 时间）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
EOF
echo "graph-growth.md 已创建"
```

Expected: 无错误输出

- [ ] **Step 3: 创建 retrieval-extraction.md 骨架**

```bash
cat > /workspace/project/LatentMind/docs/research/hippo/retrieval-extraction.md << 'EOF'
# Hippo Flow Matching 检索提取（FM Retrieval & Extraction）

> **方向**：方向 3 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)

---

## 0. 核心问题

FM 的 ODE 可逆性如何用于检索？ODE 方向如何选择？Top-K 精度与速度权衡？碎片化关联如何提取？

---

## 1. 研究目标

基于 Flow Matching 的检索模块：
- Recall@K ≥ 0.9
- 端侧 < 5ms
- 支持模糊跨域联想

---

## 2. 最小验证单元

`FMRetrieval` 模块：
- (a) 输入 query 向量
- (b) 输出 Top-K 码字 + 关联 KV
- (c) 支持 Slerp 模糊联想

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| Recall@K | Top-K 命中率 | ≥ 0.9 |
| 检索延迟 | 端到端 query → result | < 5ms |
| 联想质量 | Slerp 跨码字解码合理性 | 人工评估 + 自动 |
| 失败降级率 | 检索失败时静默退化率 | 100% |

---

## 4. 前置依赖

- [`sadko/hippo-phase0-manual.md`](../sadko/hippo-phase0-manual.md)（FM+FSQ 基础）
- 可选：[`memory-architecture.md`](./memory-architecture.md) 的码字 schema
- 可选：[`graph-growth.md`](./graph-growth.md) 的图查询 API

---

## 5. 待填充内容（实施时）

- [ ] FM 检索算法（ODE 方向选择策略：query → 码字 vs 码字 → query）
- [ ] Slerp 联想测试（跨码字解码为连续向量）
- [ ] 失败降级协议（空 KV + 全 0 嵌入的静默降级路径）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
EOF
echo "retrieval-extraction.md 已创建"
```

Expected: 无错误输出

- [ ] **Step 4: 验证三个骨架文件创建成功**

```bash
cd /workspace/project/LatentMind
for f in memory-architecture.md graph-growth.md retrieval-extraction.md; do
  [ -f "docs/research/hippo/$f" ] && echo "OK: $f 存在" || echo "FAIL: $f 缺失"
  grep -q "## 0. 核心问题" "docs/research/hippo/$f" && echo "OK: $f §0 存在" || echo "FAIL: $f §0 缺失"
  grep -q "## 6. 鲁棒性评分" "docs/research/hippo/$f" && echo "OK: $f §6 存在" || echo "FAIL: $f §6 缺失"
done
```

Expected: 9 行 `OK: ...` 输出（3 文件 × 3 检查）

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/hippo/memory-architecture.md docs/research/hippo/graph-growth.md docs/research/hippo/retrieval-extraction.md
git commit -m "feat(hippo): 创建三子方向骨架文件 (memory/graph/retrieval)"
```

---

### Task 6: 创建 sadko/README.md 和 logos/README.md（子目录索引）

**Files:**
- Create: `docs/research/sadko/README.md`
- Create: `docs/research/logos/README.md`

- [ ] **Step 1: 写 sadko/README.md（含 SADKO 主干研究索引）**

```bash
cat > /workspace/project/LatentMind/docs/research/sadko/README.md << 'EOF'
# SADKO 主干研究（SADKO Main Research）

> **定位**：感知 + 记忆 + 知识 + 多模态
> **架构**：异构双脑（左脑 AR + 右脑 Hippo）+ 扩散桥梁
> **状态**：2026-07-29 战略升级后固化（详见 [whitepaper.md](./whitepaper.md)）

---

## 1. 文件清单

### 1.1 架构设计

| 文件 | 内容 |
|------|------|
| [whitepaper.md](./whitepaper.md) | SADKO 架构设计与演进白皮书：异构双脑 + 扩散桥梁，从 64M 到 70B |
| [multimodal-native.md](./multimodal-native.md) | SADKO 多模态原生设计：2026-07-29 增补，右脑与流形数学同构 |
| [open-issues.md](./open-issues.md) | 文档体系审查：5 项技术冲突 + 多项数值/表述问题 |

### 1.2 64M 工程落地

| 文件 | 内容 |
|------|------|
| [64m-validation-plan.md](./64m-validation-plan.md) | SADKO-Native-64M 三阶段验证计划（10 周，单卡 3090）|

### 1.3 分阶段架构设计

| 文件 | 内容 |
|------|------|
| [v1-architecture.md](./v1-architecture.md) | Split-GQA + 异构 RoPE + Dual-Path FFN + CA 骨架预埋 |
| [v2-architecture.md](./v2-architecture.md) | Shared MemPool + 层专属 Cross-Attention + 压缩触发器 |
| [v3-architecture.md](./v3-architecture.md) | Hippo-Lite + FSQ + 内容寻址 Router + 动态门控 + 扩散对齐 |

### 1.4 右脑 Hippo 核心设计（已固化，与 Hippo 独立研究线的关系见各文件）

| 文件 | 内容 |
|------|------|
| [hippo-vs-gdm-review.md](./hippo-vs-gdm-review.md) | FM vs 图扩散六维裁决 |
| [hippo-graph-emergence.md](./hippo-graph-emergence.md) | 图结构涌现机制 |
| [hippo-phase0-manual.md](./hippo-phase0-manual.md) | Phase 0 执行手册 |
| [hippo-lifecycle.md](./hippo-lifecycle.md) | 右脑生命周期管理 |

---

## 2. 与其他研究线的关系

- **与 Logos 主线**：双轨分工（Logos = 推理 + 决策；SADKO = 感知 + 记忆 + 知识 + 多模态）
- **与 Hippo 独立研究线**：[`../hippo/`](../hippo/) 是 SADKO 右脑关键技术的独立延伸（自 2026-07-30 起），不复用本目录的 64M 验证计划

---

## 3. 维护规范

1. 单主题一个文件，并在本 README §1 登记
2. 跨文档引用使用相对路径（`./`、`../hippo/`、`../logos/`）
3. 语言：中文为主，英文技术术语保留
EOF
echo "sadko/README.md 已创建"
```

Expected: 无错误输出

- [ ] **Step 2: 写 logos/README.md（含 Logos 主线研究索引）**

```bash
cat > /workspace/project/LatentMind/docs/research/logos/README.md << 'EOF'
# Logos 主线研究（Logos Main Research）

> **定位**：推理 + 决策 + 端侧哲学
> **架构**：分层双时间尺度循环（H/L 借鉴 HRM-Text，独立设计）
> **状态**：2026-07-29 战略升级后固化（详见 [whitepaper.md](./whitepaper.md)）

---

## 1. 文件清单

### 1.1 架构设计

| 文件 | 内容 |
|------|------|
| [whitepaper.md](./whitepaper.md) | Logos 主线架构白皮书：循环即推理 + 多轨迹即决策 + 端侧哲学 |
| [k-strategy.md](./k-strategy.md) | K 值策略与端侧可行性：K=2 默认 + per-token early exit + PLT/Radix Cache |
| [roadmap.md](./roadmap.md) | 主线路线图（+0 到 +12 月）|

### 1.2 64M 工程落地

| 文件 | 内容 |
|------|------|
| [64m-validation-plan.md](./64m-validation-plan.md) | Logos-Native-64M 三阶段验证计划（10 周，单卡 3090）|

### 1.3 分阶段架构设计

| 文件 | 内容 |
|------|------|
| [v1-architecture.md](./v1-architecture.md) | v1.0 架构设计：四种双时间尺度实现 |
| [v2-architecture.md](./v2-architecture.md) | v2.0 架构设计：五种循环策略对比 |
| [v3-architecture.md](./v3-architecture.md) | v3.0 架构设计：GRAM + Radix Cache 多路径 |

### 1.4 双轨协调

| 文件 | 内容 |
|------|------|
| [sadko-64m-coordination.md](./sadko-64m-coordination.md) | Logos / SADKO 双轨 64M 协调中枢（含 Phase 0 共享前置）|

---

## 2. 与其他研究线的关系

- **与 SADKO 主干**：双轨分工（Logos = 推理 + 决策；SADKO = 感知 + 记忆 + 知识 + 多模态）
- **与 Hippo 独立研究线**：[`../hippo/`](../hippo/) 是 SADKO 右脑关键技术的独立延伸，但 Logos 主线不直接依赖 Hippo

---

## 3. 维护规范

1. 单主题一个文件，并在本 README §1 登记
2. 跨文档引用使用相对路径
3. 语言：中文为主，英文技术术语保留
EOF
echo "logos/README.md 已创建"
```

Expected: 无错误输出

- [ ] **Step 3: 验证两个子目录 README.md 创建成功**

```bash
cd /workspace/project/LatentMind
for subdir in sadko logos; do
  [ -f "docs/research/$subdir/README.md" ] && echo "OK: $subdir/README.md 存在" || echo "FAIL: $subdir/README.md 缺失"
  grep -q "## 1. 文件清单" "docs/research/$subdir/README.md" && echo "OK: $subdir §1 存在" || echo "FAIL: $subdir §1 缺失"
done
```

Expected: 4 行 `OK: ...` 输出（2 子目录 × 2 检查）

- [ ] **Step 4: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/sadko/README.md docs/research/logos/README.md
git commit -m "docs(research): 创建 sadko/ 和 logos/ 子目录索引 README.md"
```

---

### Task 7: 重写顶级 docs/research/README.md 为三研究线索引

**Files:**
- Modify: `docs/research/README.md`（重写）

- [ ] **Step 1: 备份当前 README.md（用于参考）**

```bash
cd /workspace/project/LatentMind
cp docs/research/README.md /tmp/research-readme-backup.md
echo "已备份原 README.md 到 /tmp/research-readme-backup.md"
wc -l /tmp/research-readme-backup.md
```

Expected: 输出原 README.md 的行数（之前看到 README.md 有 200+ 行）

- [ ] **Step 2: 写新的顶级 README.md（三研究线索引）**

```bash
cat > /workspace/project/LatentMind/docs/research/README.md << 'EOF'
# 内部研究记录索引（docs/research/）

> **组织形式**：三研究线（Logos / SADKO / Hippo）+ 顶级索引
> **战略定位**：与项目"双轨 + 探索分支"哲学一致
> **最后更新**：2026-07-30（重构为三子目录结构）

---

## 0. 三研究线战略定位

| 研究线 | 子目录 | 定位 | 架构 | 状态 |
|--------|--------|------|------|------|
| **Logos 主线** | [`logos/`](./logos/) | 推理 + 决策 + 端侧哲学 | 分层双时间尺度循环（H/L）| ✅ 2026-07-29 升级 |
| **SADKO 主干** | [`sadko/`](./sadko/) | 感知 + 记忆 + 知识 + 多模态 | 异构双脑（AR + Hippo）+ 扩散桥梁 | ✅ 2026-07-29 升级 |
| **Hippo 独立研究线** | [`hippo/`](./hippo/) | 右脑关键技术独立验证（记忆 / KG / 检索）| 复用 Hippo 机制（双向 + FM + FSQ）| 🆕 2026-07-30 启动 |

**双轨 + 探索分支**：
- Logos + SADKO = 项目双轨分工（详见 [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策)）
- Hippo = SADKO 右脑关键技术的独立延伸（详见 [hippo/README.md](./hippo/README.md)）

---

## 1. 目录组织

```
docs/research/
├── README.md              # 本文件（顶级三研究线索引）
├── REFACTOR_MAP.md        # 2026-07-30 重构时的引用清单（实施产物）
│
├── sadko/                 # SADKO 主干研究
│   ├── README.md
│   ├── whitepaper.md
│   └── ... (其他 10 份)
│
├── logos/                 # Logos 主线研究
│   ├── README.md
│   ├── whitepaper.md
│   └── ... (其他 7 份)
│
└── hippo/                    # Hippo 独立研究线
    ├── README.md          # 含胼胝体接口契约
    ├── memory-architecture.md
    ├── graph-growth.md
    └── retrieval-extraction.md
```

---

## 2. 各研究线入口

- **Logos 主线**：见 [`logos/README.md`](./logos/README.md)
- **SADKO 主干**：见 [`sadko/README.md`](./sadko/README.md)
- **Hippo 独立研究线**：见 [`hippo/README.md`](./hippo/README.md)（含胼胝体接口契约五元组）

---

## 3. 跨研究线协调文档

- 双轨协调中枢：[`logos/sadko-64m-coordination.md`](./logos/sadko-64m-coordination.md)
- Phase 0 实施指南（外部）：[`../implementation/phase-0-implementation-guide.md`](../implementation/phase-0-implementation-guide.md)
- Phase 0 失败回退 SOP（外部）：[`../implementation/phase-0-recovery-sop.md`](../implementation/phase-0-recovery-sop.md)

---

## 4. 维护规范

1. **新增研究记录**：单主题一个文件，并在所属子目录的 README.md 登记
2. **跨子目录引用**：使用相对路径（`./`、`../sadko/`、`../logos/`、`../hippo/`）
3. **跨研究线协调文档**：放 logos/ 下作为 Logos 主导的协调（如 logos/sadko-64m-coordination.md）
4. **接口契约变更**：Hippo 研究线的胼胝体契约变更需双侧 review（详见 [hippo/README.md §2.3](./hippo/README.md#23-关键不变量)）
5. **语言**：中文为主，英文技术术语保留
6. **目录重构记录**：本次重构（2026-07-30）详见 [REFACTOR_MAP.md](./REFACTOR_MAP.md)
EOF
echo "顶级 docs/research/README.md 已重写"
```

Expected: 无错误输出

- [ ] **Step 3: 验证顶级 README.md 包含三研究线入口**

```bash
cd /workspace/project/LatentMind
grep -q "Logos 主线" docs/research/README.md && echo "OK: Logos 入口存在" || echo "FAIL: Logos 入口缺失"
grep -q "SADKO 主干" docs/research/README.md && echo "OK: SADKO 入口存在" || echo "FAIL: SADKO 入口缺失"
grep -q "Hippo 独立研究线" docs/research/README.md && echo "OK: Hippo 入口存在" || echo "FAIL: Hippo 入口缺失"
grep -q "REFACTOR_MAP" docs/research/README.md && echo "OK: REFACTOR_MAP 引用存在" || echo "FAIL: REFACTOR_MAP 引用缺失"
```

Expected: 4 行 `OK: ...` 输出

- [ ] **Step 4: 验证目录结构最终形态**

```bash
cd /workspace/project/LatentMind
echo "=== 顶级目录 ==="
ls docs/research/
echo ""
echo "=== sadko/ 子目录 ==="
ls docs/research/sadko/
echo ""
echo "=== logos/ 子目录 ==="
ls docs/research/logos/
echo ""
echo "=== hippo/ 子目录 ==="
ls docs/research/hippo/
```

Expected: 顶级目录含 README.md + REFACTOR_MAP.md + sadko/ + logos/ + hippo/；每个子目录含 README.md 和具体文档

- [ ] **Step 5: Commit**

```bash
cd /workspace/project/LatentMind
git add docs/research/README.md
git commit -m "docs(research): 重写顶级 README.md 为三研究线索引"
```

---

### Task 8: 更新 AGENTS.md 引用路径

**Files:**
- Modify: `AGENTS.md`（多处硬编码路径更新）

- [ ] **Step 1: 扫描 AGENTS.md 中所有 docs/research/ 引用**

```bash
cd /workspace/project/LatentMind
grep -n "docs/research/" AGENTS.md
```

Expected: 输出 AGENTS.md 中所有 docs/research/ 引用位置（行号 + 内容）

- [ ] **Step 2: 更新 AGENTS.md 中 logos-* 引用**

```bash
cd /workspace/project/LatentMind
# AGENTS.md 中可能引用 docs/research/logos-*.md
sed -i 's|docs/research/logos-whitepaper\.md|docs/research/logos/whitepaper.md|g; s|docs/research/logos-k-strategy\.md|docs/research/logos/k-strategy.md|g; s|docs/research/logos-roadmap\.md|docs/research/logos/roadmap.md|g; s|docs/research/logos-64m-validation-plan\.md|docs/research/logos/64m-validation-plan.md|g; s|docs/research/logos-v1-architecture\.md|docs/research/logos/v1-architecture.md|g; s|docs/research/logos-v2-architecture\.md|docs/research/logos/v2-architecture.md|g; s|docs/research/logos-v3-architecture\.md|docs/research/logos/v3-architecture.md|g; s|docs/research/logos-sadko-64m-coordination\.md|docs/research/logos/sadko-64m-coordination.md|g' AGENTS.md
echo "AGENTS.md logos 引用更新完成"
```

Expected: 无错误输出

- [ ] **Step 3: 更新 AGENTS.md 中 sadko-* 引用**

```bash
cd /workspace/project/LatentMind
sed -i 's|docs/research/sadko-whitepaper\.md|docs/research/sadko/whitepaper.md|g; s|docs/research/sadko-64m-validation-plan\.md|docs/research/sadko/64m-validation-plan.md|g; s|docs/research/sadko-v1-architecture\.md|docs/research/sadko/v1-architecture.md|g; s|docs/research/sadko-v2-architecture\.md|docs/research/sadko/v2-architecture.md|g; s|docs/research/sadko-v3-architecture\.md|docs/research/sadko/v3-architecture.md|g; s|docs/research/sadko-multimodal-native\.md|docs/research/sadko/multimodal-native.md|g; s|docs/research/sadko-open-issues\.md|docs/research/sadko/open-issues.md|g; s|docs/research/sadko-hippo-vs-gdm-review\.md|docs/research/sadko/hippo-vs-gdm-review.md|g; s|docs/research/sadko-hippo-graph-emergence\.md|docs/research/sadko/hippo-graph-emergence.md|g; s|docs/research/sadko-hippo-phase0-manual\.md|docs/research/sadko/hippo-phase0-manual.md|g; s|docs/research/sadko-hippo-lifecycle\.md|docs/research/sadko/hippo-lifecycle.md|g' AGENTS.md
echo "AGENTS.md sadko 引用更新完成"
```

Expected: 无错误输出

- [ ] **Step 4: 验证 AGENTS.md 中无旧路径残留**

```bash
cd /workspace/project/LatentMind
grep -E "docs/research/(sadko-|logos-)[a-z]" AGENTS.md || echo "OK: AGENTS.md 无旧前缀引用残留"
```

Expected: `OK: AGENTS.md 无旧前缀引用残留`

- [ ] **Step 5: 检查并更新其他可能引用 docs/research/ 的文件**

```bash
cd /workspace/project/LatentMind
# 扫描所有可能引用 docs/research/ 的文件
grep -rln "docs/research/" docs/ README.md 2>/dev/null | grep -v "docs/research/REFACTOR_MAP" | grep -v "docs/research/README" | grep -v "docs/research/sadko/" | grep -v "docs/research/logos/" | grep -v "docs/research/hippo/"
```

Expected: 输出任何引用 docs/research/ 旧路径的文件清单（如有，逐个处理）

- [ ] **Step 6: 如果 Step 5 有输出，对每个文件执行类似 sed 替换**

```bash
cd /workspace/project/LatentMind
# 对 docs/architecture.md 和 docs/implementation/README.md 做类似更新
# 注意：相对路径引用（如 ../research/sadko-*）需要 ../research/sadko/
for f in docs/architecture.md docs/implementation/README.md; do
  if [ -f "$f" ]; then
    sed -i 's|sadko-whitepaper\.md|sadko/whitepaper.md|g; s|sadko-64m-validation-plan\.md|sadko/64m-validation-plan.md|g; s|sadko-v1-architecture\.md|sadko/v1-architecture.md|g; s|sadko-v2-architecture\.md|sadko/v2-architecture.md|g; s|sadko-v3-architecture\.md|sadko/v3-architecture.md|g; s|sadko-multimodal-native\.md|sadko/multimodal-native.md|g; s|sadko-open-issues\.md|sadko/open-issues.md|g; s|sadko-hippo-vs-gdm-review\.md|sadko/hippo-vs-gdm-review.md|g; s|sadko-hippo-graph-emergence\.md|sadko/hippo-graph-emergence.md|g; s|sadko-hippo-phase0-manual\.md|sadko/hippo-phase0-manual.md|g; s|sadko-hippo-lifecycle\.md|sadko/hippo-lifecycle.md|g; s|logos-whitepaper\.md|logos/whitepaper.md|g; s|logos-k-strategy\.md|logos/k-strategy.md|g; s|logos-roadmap\.md|logos/roadmap.md|g; s|logos-64m-validation-plan\.md|logos/64m-validation-plan.md|g; s|logos-v1-architecture\.md|logos/v1-architecture.md|g; s|logos-v2-architecture\.md|logos/v2-architecture.md|g; s|logos-v3-architecture\.md|logos/v3-architecture.md|g; s|logos-sadko-64m-coordination\.md|logos/sadko-64m-coordination.md|g' "$f"
    echo "已更新 $f"
  fi
done
```

Expected: 输出 "已更新 docs/architecture.md" 和 "已更新 docs/implementation/README.md"（如文件存在）

- [ ] **Step 7: Commit**

```bash
cd /workspace/project/LatentMind
git add AGENTS.md docs/architecture.md docs/implementation/README.md 2>/dev/null
git status --short
git commit -m "docs(refactor): 更新 AGENTS.md 等外部引用为新子目录路径"
```

---

### Task 9: 最终验证 — 跨文档引用一致性检查

**Files:**
- (验证任务，无文件创建)

- [ ] **Step 1: 全项目扫描确认无旧路径引用**

```bash
cd /workspace/project/LatentMind
# 排除新结构中的合法引用（sadko/、logos/、hippo/ 子目录内的）
grep -rn "docs/research/sadko-\|docs/research/logos-" . --include="*.md" 2>/dev/null | grep -v "docs/research/sadko/sadko-64m-coordination.md" | grep -v "docs/research/logos/sadko-64m-coordination.md" || echo "OK: 全项目无旧前缀引用残留"
```

Expected: `OK: 全项目无旧前缀引用残留`

- [ ] **Step 2: 验证所有子目录文件存在**

```bash
cd /workspace/project/LatentMind
echo "=== sadko/ 子目录文件数 ==="
ls docs/research/sadko/*.md | wc -l
echo ""
echo "=== logos/ 子目录文件数 ==="
ls docs/research/logos/*.md | wc -l
echo ""
echo "=== hippo/ 子目录文件数 ==="
ls docs/research/hippo/*.md | wc -l
echo ""
echo "=== 顶级 docs/research/ 文件数（应仅含 README.md + REFACTOR_MAP.md）==="
ls docs/research/*.md | wc -l
ls docs/research/*.md
```

Expected:
- sadko/：12 个 .md 文件（11 份迁移 + 1 个 README）
- logos/：9 个 .md 文件（8 份迁移 + 1 个 README）
- hippo/：4 个 .md 文件（README + 3 个骨架）
- 顶级：2 个文件（README.md + REFACTOR_MAP.md）

- [ ] **Step 3: 检查跨子目录链接的可解析性（spot check）**

```bash
cd /workspace/project/LatentMind
# 抽查 5 个跨子目录引用是否目标文件存在
echo "=== Hippo README 引用的 sadko/hippo-* 文档 ==="
grep -E "\.\./sadko/hippo-[a-z-]+\.md" docs/research/hippo/README.md | head -3 | while read line; do
  target=$(echo "$line" | grep -oE "\.\./sadko/hippo-[a-z-]+\.md")
  full_path="docs/research/hippo/$target"
  full_path=$(echo "$full_path" | sed 's|\.\./|../|g')
  [ -f "$full_path" ] && echo "OK: $target 存在" || echo "FAIL: $target 缺失"
done

echo ""
echo "=== 顶级 README 引用的子目录 README ==="
grep -E "\./(sadko|logos|hippo)/README\.md" docs/research/README.md | while read line; do
  target=$(echo "$line" | grep -oE "\./(sadko|logos|hippo)/README\.md")
  full_path="docs/research/$target"
  [ -f "$full_path" ] && echo "OK: $target 存在" || echo "FAIL: $target 缺失"
done
```

Expected: 5 行 `OK: ...` 输出（跨子目录引用全部存在）

- [ ] **Step 4: 生成最终目录结构报告**

```bash
cd /workspace/project/LatentMind
echo "=== docs/research/ 最终结构 ==="
tree docs/research/ -L 2 2>/dev/null || find docs/research/ -maxdepth 2 -type f | sort
```

Expected: 显示完整三子目录结构

- [ ] **Step 5: Commit（REFACTOR_MAP.md 标记为最终状态）**

```bash
cd /workspace/project/LatentMind
# 在 REFACTOR_MAP.md 末尾追加验证完成标记
echo "" >> docs/research/REFACTOR_MAP.md
echo "## 验证完成" >> docs/research/REFACTOR_MAP.md
echo "" >> docs/research/REFACTOR_MAP.md
echo "2026-07-30 验证完成：所有跨文档引用已更新为新子目录路径。" >> docs/research/REFACTOR_MAP.md
git add docs/research/REFACTOR_MAP.md
git commit -m "docs(research): REFACTOR_MAP.md 标记验证完成"
```

---

## 验收标准

完成所有 Tasks 后，必须满足：

- [ ] `docs/research/sadko/` 含 11 份原文档 + 1 个 README.md（12 个 .md 文件）
- [ ] `docs/research/logos/` 含 8 份原文档 + 1 个 README.md（9 个 .md 文件）
- [ ] `docs/research/hippo/` 含 1 个 README.md（含胼胝体契约）+ 3 个骨架文件
- [ ] 顶级 `docs/research/` 仅含 README.md + REFACTOR_MAP.md
- [ ] `hippo/README.md` §2 含完整的接口契约五元组 + 五个不变量
- [ ] AGENTS.md 中所有 docs/research/ 引用已更新
- [ ] 全项目 grep 无旧前缀（sadko-*.md / logos-*.md）引用残留
- [ ] 所有 git commit 信息清晰、原子化

## 下一步

完成本 plan 后：

1. 为三个子方向（memory-architecture / graph-growth / retrieval-extraction）各生成独立 plan
2. 启动三子方向的最小验证单元（按 hippo/README.md §3 验证哲学）
3. 通过胼胝体接口契约（§2）逐步把成果接回 SADKO 主干（T+3 阶段）