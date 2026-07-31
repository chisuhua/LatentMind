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
