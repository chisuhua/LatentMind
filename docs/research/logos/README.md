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
- **与 ELF 独立研究线**：[`../elf/`](../elf/) 是 SADKO 右脑关键技术的独立延伸，但 Logos 主线不直接依赖 ELF

---

## 3. 维护规范

1. 单主题一个文件，并在本 README §1 登记
2. 跨文档引用使用相对路径
3. 语言：中文为主，英文技术术语保留
