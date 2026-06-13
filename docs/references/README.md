# 参考资料索引（docs/references/）

> **用途**：集中存放 LatentMind 项目所依赖的**外部研究/项目元信息**，作为开发参考
> **最后更新**：2026-06-13

---

## 1. 文件清单

| 文件 | 内容 | 优先级 | 状态 |
|---|---|---|---|
| [hrm-text.md](./hrm-text.md) | HRM-Text 论文/模型完整笔记（Sapient 2026-05） | 🔴 核心 | ✅ |
| [gram.md](./gram.md) | GRAM 论文笔记（v1.5 集成关键） | 🔴 核心 | ✅ v2 增强版 |
| [projects.md](./projects.md) | 相关项目元信息（ChipForge / HydraForge / AgenticLlama / minimind） | 🔴 核心 | ✅ |
| [hrm-original.md](./hrm-original.md) | HRM 原始 27M 论文笔记（MagicNorm 概念溯源） | 🟡 参考 | ✅ |
| [trm.md](./trm.md) | TRM 论文笔记（"Less is More" 直接挑战 HRM）| 🟡 重要 | ✅ |
| [rrm-survey.md](./rrm-survey.md) | RRM-B 谱系调查（TRM / Looped TF / Hyperloop / SE-RRM / Huginn / Ouro + 下一代架构）| 🟡 重要 | ✅ |
| [rrm-reward.md](./rrm-reward.md) | RRM-A 笔记（清华 2505.14674，奖励推理模型）| 🟡 参考 | ✅ |
| [papers/](./papers/) | 论文源文件（PDF/HTML） | 🔴 核心 | ✅ |
| [papers/README.md](./papers/README.md) | 论文源文件清单 + 下载说明 | — | ✅ |
| `huginn.md` | Huginn 3.5B 笔记（latent recurrent 对比） | 🟢 对比 | ⏳ 合并到 rrm-survey.md §1.5 |
| `ouro.md` | Ouro 1.4B 笔记（looped 对比） | 🟢 对比 | ⏳ 合并到 rrm-survey.md §1.5 |

> 📦 **新增目录**：[../rfcs/](../rfcs/) 存放**内部 R&D 设计提案**（GRR-300M、MR-300M），不属于参考资料。

---

## 2. 维护规范

### 2.1 添加新参考资料的流程

1. **确认优先级**：核心依赖（必做）/ 工程参考（按需）/ 学术对比（可选）
2. **资料来源要求**：
   - 优先官方源（论文 arXiv、官方博客、HF 模型卡、GitHub README）
   - 引用时附上完整 URL
   - 关键数字直接引自原文（带出处）
3. **结构要求**：
   - 一句话定位（开头）
   - 链接速查表
   - 核心创新 / 架构 / 训练 / 关键数字
   - 对 LatentMind 的复用价值与风险
4. **语言**：中文为主，英文技术术语保留

### 2.2 文件命名规范

```
references/
├── <技术名>.md          # 单技术笔记
├── projects.md          # 关系图与项目卡片（唯一）
├── papers/              # 论文源文件
│   ├── arxiv-<YYMM>.<NNNN>-<slug>.pdf
│   ├── arxiv-<YYMM>.<NNNN>-<slug>.html
│   └── README.md
└── README.md            # 本文件（索引）
```

### 2.3 引用格式

- 论文：`作者. *标题*. arXiv:NNNN.NNNNN（YYYY-MM）`
- 博客：`机构. *标题*. URL（YYYY-MM-DD）`
- 代码：`仓库. 文件路径. URL`
- 模型：`仓库名. 许可证. URL`

### 2.4 更新触发条件

| 触发 | 动作 |
|---|---|
| 新论文/新版本发布 | 更新对应笔记 + 本索引 + papers/ 子目录 |
| 权重/许可证变化 | 更新"开源权重"章节 |
| 集成时发现新风险 | 在对应笔记追加"复现风险"小节 |
| 项目卡片信息变化 | 更新 `projects.md` |

---

## 3. 速查入口

### 3.1 我想了解 HRM-Text 的什么 → 去哪查

| 问题 | 查 [hrm-text.md](./hrm-text.md) 哪一节 |
|---|---|
| 论文在哪？许可证？ | §2 链接速查 |
| 三大创新点？ | §3 核心创新 |
| 1B 模型的具体配置？ | §4 架构参数 |
| 训练用什么优化器？多少 H100？多久？ | §5 训练配方 |
| 训练数据从哪来？ | §6 数据集组成 |
| 性能 vs 同类模型？ | §7 基准结果 |
| 权重怎么下载？ | §8 开源权重 |
| GitHub 仓库结构？关键文件？ | §9 仓库结构 |
| 多模态复用有什么坑？ | §10 复现风险 |
| 我该怎么用？ | §11 复用建议 |

### 3.2 我想了解 HRM（原始 27M）的什么 → 去哪查

| 问题 | 查 [hrm-original.md](./hrm-original.md) 哪一节 |
|---|---|
| 为什么有"分层循环"概念？ | §3 核心创新 |
| 27M 模型具体怎么训？ | §4 实验结果 |
| 跟 HRM-Text 怎么演进？ | §5 与 HRM-Text 关系 |

### 3.3 我想了解 TRM 怎么"挑战" HRM → 去哪查

| 问题 | 查 [trm.md](./trm.md) 哪一节 |
|---|---|
| TRM 比 HRM 强在哪？ | §3 核心创新 / §4 关键结果 |
| 我要不要重写 v1.0 backbone？ | §5 对 LatentMind 的启示 |
| 跟 GRAM 怎么对比？ | §5.4 与 GRAM 的对比 |
| 我该做什么决策？ | §7 对项目决策的影响 |

### 3.4 我想了解上下游项目 → 去哪查

| 问题 | 查 [projects.md](./projects.md) 哪一节 |
|---|---|
| LatentMind 在生态中位置？ | §1 上下游关系图 |
| 每个项目做什么？路径在哪？ | §2 项目卡片 |
| 谁依赖谁？ | §3 依赖矩阵 |
| 哪些资源可跨项目复用？ | §4 共享资源 |
| 哪些参考材料还没收集？ | §5 研究资料收集状态 |

### 3.5 我想了解 RRM 谱系全景 → 去哪查

| 问题 | 查 [rrm-survey.md](./rrm-survey.md) 哪一节 |
|---|---|
| TRM / Looped TF / Hyperloop / SE-RRM / Huginn / Ouro 简介 | §1 RRM-B 谱系 |
| Google Titans / Liquid AI / SpikingBrain / 商汤 NEO-Unify | §2 下一代架构 |
| 架构横向对比 | §3 核心架构横向对比 |
| 与 LatentMind 决策相关 | §5 与 LatentMind 决策相关 |

### 3.6 我想了解 RRM 奖励模型（清华 RRM-A）→ 去哪查

| 问题 | 查 [rrm-reward.md](./rrm-reward.md) 哪一节 |
|---|---|
| RRM-A 和 RRM-B 的区别？ | 顶部"重要命名歧义" |
| 与传统 Scalar RM 区别？ | §2 对比表 |
| 开源项目清单？ | §4 开源项目与资源 |
| 与 LatentMind 关系？ | §6 与 LatentMind 项目的关系 |

### 3.7 我想查看论文原文 PDF/HTML → 去哪查

| 问题 | 查 [papers/README.md](./papers/README.md) 哪一节 |
|---|---|
| 哪些论文有源文件？ | §1 文件清单 |
| 命名规范？ | §2 文件命名规范 |
| 下载来源？ | §3 下载来源说明 |
| HRM-Text PDF 为什么没有？ | §3.3 / §5 重试策略 |
| 怎么读 PDF / HTML？ | §4 使用方式 |

---

## 4. 收集策略

### 4.1 已收集（外部源）

- ✅ HRM-Text 论文（arXiv:2605.20613）— abs 摘要页 + 详细笔记
- ✅ HRM-Text HF 模型卡
- ✅ HRM-Text GitHub 仓库结构
- ✅ HRM-Text HF Transformers PR（#46025）
- ✅ HRM-Text vLLM PR（#43098）
- ✅ HRM-Text MLX-VLM PR（#1238）
- ✅ GRAM 论文（arXiv:2605.19376）— PDF + HTML
- ✅ GRAM 项目页（https://ahn-ml.github.io/gram-website）
- ✅ HRM 原始 27M 论文（arXiv:2506.21734）— PDF + HTML
- ✅ TRM 论文（arXiv:2510.04871）— PDF + HTML
- ✅ TRM GitHub 仓库（待补充链接）

### 4.2 待补充（外部源）

- ⏳ Ouro 1.4B（arXiv ID 待查）
- ⏳ Huginn 3.5B（arXiv ID 待查）
- ⏳ HRM-MoE 仓库（GitHub: XiaoYee/HRM-MoE，无 paper）
- ⏳ MagicNorm 早期引用（HRM 论文内的概念，无独立 paper）
- ⏳ AdamATan2 优化器原文（HRM-Text 论文 [20] 引用）

### 4.3 待补充（内部源）

- ⏳ ChipForge 硬件规格（量化约束来源）
- ⏳ AgenticLlama 算子清单（PrefixLM attention 支持情况）
- ⏳ HydraForge 调度接口（推理结果如何被消费）
- ⏳ minimind 与 LatentMind 的 tokenizer / 数据 pipeline 复用点

---

## 5. 工具说明

参考资料的内容来自两种途径：

1. **本会话内嵌**（已写入文件的内容）：可直接 `cat` 查阅
2. **外部 URL 链接**（仅记录链接）：需要联网时再次获取
   - 建议用 `webfetch` 工具拉取
   - 用 `librarian` 子 agent 做系统研究（参见 [AGENTS.md](../../AGENTS.md)）
3. **papers/ 源文件**（PDF/HTML）：可离线查阅

---

**最后更新**：2026-06-13
