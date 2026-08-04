# Logos 主线研究（Logos Main Research）

> **定位**：推理 + 决策 + 端侧哲学
> **架构**：分层双时间尺度循环（H/L 借鉴 HRM-Text，独立设计）
> **状态**：2026-07-31 更新（新增 §3.5/§3.6/§4.6 验证计划 + PLT+Φ 集成设计）

---

## 1. 文件清单

### 1.1 架构设计

| 文件 | 内容 |
|------|------|
| [whitepaper.md](./whitepaper.md) | Logos 主线架构白皮书：循环即推理 + 多轨迹即决策 + 端侧哲学 |
| [k-strategy.md](./k-strategy.md) | K 值策略与端侧可行性（v1.5：含 §2.2 PLT 澄清 + §2.2.7 PLT+Φ 集成 + §2.6 Loop B fallback）|
| [roadmap.md](./roadmap.md) | 主线路线图（+0 到 +12 月）|

### 1.2 64M 工程落地

| 文件 | 内容 |
|------|------|
| [64m-validation-plan.md](./64m-validation-plan.md) | Logos-Native-64M 验证计划（v1.5：§3.5 DiscoLoop 探针 + §3.6 Loop B 备选 + §4.6 L-PLT + §4.6.9 L-PLT+Φ）|

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

### 1.5 跨线集成设计 🆕

| 文件 | 内容 |
|------|------|
| [codebook-config-sop.md](./codebook-config-sop.md) | FSQ 代码本配置冲突解决 SOP（[8,8,4] vs 256，Logos 消费 FSQ 前置依赖）|
| [fsq-consumption-design.md](./fsq-consumption-design.md) | Logos 消费 Hippo FSQ 条件设计（4 前置条件门控 + INV-6~9）|

---

## 2. 与其他研究线的关系

- **与 SADKO 主干**：双轨分工（Logos = 推理 + 决策；SADKO = 感知 + 记忆 + 知识 + 多模态）
- **与 Hippo 独立研究线**：[`../hippo/`](../hippo/) 是 SADKO 右脑关键技术的独立延伸；Logos 通过 [fsq-consumption-design.md](./fsq-consumption-design.md) 条件消费 Hippo FSQ
- **与 Thumos 独立研究线**：[`../thumos/`](../thumos/) 的内化编排作用在 Logos 主干之上
- **外部研究参考**：[discoloop.md](../../references/discoloop.md)（§3.5 探针）、[streaming-llm.md](../../references/streaming-llm.md) + [inf-llm.md](../../references/inf-llm.md)（§3.6 备选）、[loopcoder-v2.md](../../references/loopcoder-v2.md)（PLT）

---

## 3. 维护规范

1. 单主题一个文件，并在本 README §1 登记
2. 跨文档引用使用相对路径
3. 语言：中文为主，英文技术术语保留
