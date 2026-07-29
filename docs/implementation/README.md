# 实施文档索引（docs/implementation/）

> **用途**：存放**近期可执行**的工程规范、实施指南、脚本模板、SOP——区别于 `docs/research/`（总体规划与架构设计）和 `docs/rfcs/`（R&D 设计提案）。
> **最后更新**：2026-07-29

---

## 与其它文档目录的区分

| 目录 | 内容 | 时间范围 | 可执行性 |
|------|------|----------|---------|
| `docs/research/` | 架构白皮书、路线图、协调文档 | 长期（+0 至 +12 月） | ❌ 规划/设计 |
| `docs/rfcs/` | 内部 R&D 设计提案 | 中期（+0 至 +6 月） | ❌ 提案/草案 |
| **`docs/implementation/`** | **实施指南、脚本、SOP、验收模板** | **近期（+0 至 +2 月）** | **✅ 可执行** |

---

## 文件清单

| 文件 | 内容 | 上游文档 |
|------|------|---------|
| [phase-0-implementation-guide.md](./phase-0-implementation-guide.md) | **Phase 0 实施指南**：6 项 P.0.x 的训练脚本、配置 YAML、监控仪表盘、错误处理 SOP、验收脚本；Phase 0 验收报告模板；通信协议 | [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md) |
| [phase-0-recovery-sop.md](./phase-0-recovery-sop.md) | **Phase 0 失败回退 SOP**：每个 P.0.x 失败决策树、紧急升级路径（L1/L2/L3）、联系人矩阵、失败案例库 | [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md) |

---

## 新增文档的规范

1. 文档必须明确标注**上游文档**（引用 `docs/research/` 中的对应设计文档）
2. 必须包含**可执行内容**——脚本、配置 YAML、SOP、验收模板（至少包含 2 样）
3. 必须标注**时效性**——"Phase 0 专用"或"Phase 1 专用"等
4. 语言：中文为主，技术术语保留英文

---

**最后更新**：2026-07-29