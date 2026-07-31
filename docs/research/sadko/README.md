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
