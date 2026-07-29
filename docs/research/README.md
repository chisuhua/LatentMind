# 内部研究记录索引（docs/research/）

> **用途**：存放 LatentMind 项目的**内部架构研究讨论记录**（设计推演、机制证伪方案、规模化策略），与 `docs/references/`（外部论文笔记）和 `docs/rfcs/`（正式 R&D 提案）区分
> **最后更新**：2026-07-29

---

## 0. LatentMind 双轨架构定位（Logos 主线 + SADKO 探索分支）

> 📌 **2026-07-29 战略升级（v1.2）**：
> - 主线正式命名为 **Logos / 逻各斯**（希腊哲学：理性之原则）— **全新架构，从零训练**
> - SADKO 路线升级为"**多模态流形记忆与感知**"核心研究方向 — **全新架构，从零训练**
> - HRM-Text / GRAM 等仅作**架构灵感参考**，不集成其权重
> - 64M 是起点，逐步扩展到 300M / 1B / 1.5B（每级独立训练）
> - K 值是超参数，不是架构决策

| 维度 | **Logos 主线** | **SADKO 探索分支** |
| :--- | :--- | :--- |
| **backbone** | 分层双时间尺度循环（H/L 借鉴 HRM-Text，独立设计）| AR-Native（标准 Decoder-only + Split-GQA，独立设计） |
| **记忆范式** | Per-Token 早退 + Radix Cache 多路径 | ELF Flow Matching + FSQ 全局语义压缩 |
| **核心哲学** | 循环即推理 + 多路径即决策 | 异构双脑（AR 之箭 + FM 之环） + 分子自组装 |
| **天然优势域** | **推理 + 决策** | **感知 + 记忆 + 知识 + 多模态** |
| **是否从零训练** | ✅ 是 | ✅ 是 |
| **是否复用外部预训练权重** | ❌ 否（HRM-Text/GRAM 仅参考）| ❌ 否 |
| **共享假设** | 压缩即智能 / 潜空间为核心状态 / 端侧推理 | 同左 |
| **互斥假设** | 单一递归骨干 + 内部循环 | 异构架构 + 跨界面对齐（扩散桥梁）|

两条路线共享"压缩即智能"的哲学基础，但骨干选择与记忆机制不同。**两条路线都是全新架构，从零训练**。规模扩展：64M → 300M → 1B → 1.5B（每级独立训练，逐步扩大）。

**关键事件**（2026-07-29 v1.2 战略调整）：
1. **logos-roadmap.md 重写**——移除 HRM-Text 集成假设，改为规模扩展（64M → 300M → 1B）
2. **logos-k-strategy.md 重定位**——K 值是超参数（不是固定最优值），不需要前置 K-sweep
3. **logos-v3-architecture.md 去除 GRAM 集成**——独立实现多路径并行，不集成 GRAM 训练流程
4. **RDD-0001 完全废弃**——不再需要前置 K-sweep
5. **docs/architecture.md v1.2 重写**——明确"全新架构，从零训练"

详细主线文档见 [logos-whitepaper.md](./logos-whitepaper.md) / [logos-k-strategy.md](./logos-k-strategy.md) / [logos-roadmap.md](./logos-roadmap.md)。

---

## 1. 目录定位

| 目录 | 内容性质 |
|---|---|
| `docs/references/` | 外部研究/论文笔记（HRM-Text、GRAM、TRM 等） |
| `docs/rfcs/` | 内部正式 R&D 设计提案（GRR-300M、MR-300M） |
| `docs/research/` | **内部研究讨论记录**（架构推演、多轮总结的合并稿） |

---

## 2. 文件清单

### 2.1 Logos 主线文档

| 文件 | 内容 | 状态 |
|---|---|---|
| [logos-whitepaper.md](./logos-whitepaper.md) | **Logos 主线架构白皮书**：循环即推理 + 多轨迹即决策 + 端侧哲学；v1.0 Demo + v1.5 完整架构蓝图；与 ChipForge APU 集成；与 SADKO 双轨分工；风险与降级；演进路径 | ✅ |
| [logos-k-strategy.md](./logos-k-strategy.md) | **Logos K 值策略与端侧可行性**：LoopCoder-v2 的主线消化 + K=2 默认策略 + per-token early exit + **PLT/HLT-PLT + Radix Cache 多路径 + 层次化推理**三方案 + 端侧时延预算 + RDD-0001 降级为 Plan B | ✅ |
| [logos-roadmap.md](./logos-roadmap.md) | **Logos 主线路线图**（+0 到 +12 月）：详细里程碑、验收标准、与 SADKO 协同、双轨分工决策时间表 | ✅ |
| [logos-64m-validation-plan.md](./logos-64m-validation-plan.md) | **Logos-Native-64M 三阶段验证计划**：v1.0 双时间尺度对比 → v2.0 多种循环策略 → v3.0 GRAM + Radix Cache，10 周单卡 3090（含 SADKO Split-GQA 借鉴 + 与 SADKO 64M 协同）| ✅ 新增 |
| [logos-v1-architecture.md](./logos-v1-architecture.md) | v1.0 架构设计：四种"双时间尺度"实现——HRM H/L、SADKO Split-GQA、混合风格、标准 Transformer；详细施工图 + 权重初始化 + 4 个对照实验 | ✅ 新增 |
| [logos-v2-architecture.md](./logos-v2-architecture.md) | v2.0 架构设计：五种循环策略对比——串行 K、PLT/HLT-PLT、Per-Token 早退、Radix Cache 多路径、层次化推理；延迟-精度 Pareto + 详细实验设计 | ✅ 新增 |
| [logos-v3-architecture.md](./logos-v3-architecture.md) | v3.0 架构设计：GRAM + Radix Cache 多路径 + 测试时 scaling 曲线；端侧多假设决策完整方案 | ✅ 新增 |

### 2.1+ 双轨协调中枢

| 文件 | 内容 | 状态 |
|---|---|---|
| [logos-sadko-64m-coordination.md](./logos-sadko-64m-coordination.md) | **Logos / SADKO 双轨 64M 协调中枢**：Phase 0 共享前置验证（~5 周）+ Phase 1 并行验证 + Phase 2 交叉验证 + Phase 3 决策矩阵；2026-07-29 关键缺口发现后新增 | ✅ 新增 |

### 2.2 SADKO 探索分支文档

| 文件 | 内容 | 状态 |
|---|---|---|
| [sadko-whitepaper.md](./sadko-whitepaper.md) | SADKO 架构设计与演进白皮书：异构双脑（左脑 AR + 右脑 ELF）+ 扩散桥梁，从 64M 机制证伪到 70B 规模化演进的全要素合并稿 | ✅ |
| [sadko-64m-validation-plan.md](./sadko-64m-validation-plan.md) | SADKO-Native-64M 三阶段验证计划：基于 MiniMind3 的 v1.0 基座适配 → v2.0 记忆压缩 → v3.0 灵魂注入工程落地方案（10 周，单卡 3090） | ✅ |
| [sadko-v1-architecture.md](./sadko-v1-architecture.md) | v1.0 架构设计：Split-GQA + 异构 RoPE + Dual-Path FFN + CA 骨架预埋，含完整代码实现与 MiniMind3 权重初始化策略 | ✅ |
| [sadko-v2-architecture.md](./sadko-v2-architecture.md) | v2.0 架构设计：Shared MemPool（MLP 压缩器）+ 层专属 Cross-Attention 读取 + 压缩触发器，192 维潜空间读写管线 | ✅ |
| [sadko-v3-architecture.md](./sadko-v3-architecture.md) | v3.0 架构设计：ELF-Lite（Flow Matching）+ FSQ + 内容寻址 Router + 动态门控 + 左脑校验 + 扩散对齐，Phase 1-5 流水线与四大实验代码 | ✅ |
| [sadko-elf-vs-gdm-review.md](./sadko-elf-vs-gdm-review.md) | 右脑技术路线裁决：SADKO（FM+双向注意力）vs GDM-v1.2（图扩散）六维对比，放弃显式 GDM 的决策依据与 v1.2 修订（事实感知模块 + 四阶段验证） | ✅ |
| [sadko-elf-graph-emergence.md](./sadko-elf-graph-emergence.md) | 图结构涌现机制：禁止显式图损失的理由、促生长训练策略（数据几何压力/拓扑软约束）、图论验证工具箱、mRNA 分子自组装范式、解压重构的物理必要性 | ✅ |
| [sadko-elf-phase0-manual.md](./sadko-elf-phase0-manual.md) | Phase 0 执行手册：FM+FSQ 收敛验证的生命体征基线、监控仪表盘、分阶段干预策略、死亡线与逃生舱、交付物清单 | ✅ |
| [sadko-elf-lifecycle.md](./sadko-elf-lifecycle.md) | 右脑生命周期管理：增量生长三模式（码字招募/构象异构/模块扩展）、结构化遗忘（受控自噬/稳态可塑性）、系统存活六维检查清单 | ✅ |
| [sadko-open-issues.md](./sadko-open-issues.md) | 🔴 文档体系审查：5 项技术冲突（A类）、8 项数值/维度错误（B类）、8 项表述不一致（C类）、10 项设计遗漏（D类）、6 个开放问题（E类）+ 处理顺序 | ⚠️ 大项已裁决，逐条落实中 |
| [sadko-multimodal-native.md](./sadko-multimodal-native.md) | SADKO 多模态原生设计：2026-07-29 增补，从第一性原理论证右脑（双向+FM+FSQ）天然契合流形感知/记忆/知识/多模态，配套主线循环/多轨迹的推理/决策分工 | ✅ 新增 |

---

## 3. SADKO 速查入口

### 3.1 架构设计（白皮书）

#### Logos 主线（推理 + 决策）

| 问题 | 查 [logos-whitepaper.md](./logos-whitepaper.md) 哪一节 |
|---|---|
| Logos 核心设计哲学（循环即推理 / 多轨迹即决策 / 端侧哲学）？ | §1 |
| Logos 完整架构蓝图（感知 + HRM + GRAM + 双流解码）？ | §2 |
| Logos 与 SADKO 的双轨分工？ | §3 |
| Logos 训练 3 阶段？ | §4 |
| Logos 与 ChipForge APU 集成？ | §5 |
| Logos 风险与降级预案？ | §6 |
| Logos 演进路径（64M 验证 → 1B 端侧 → 1.5B 完整）？ | §7 |
| K 值策略、LoopCoder-v2 应对？ | [logos-k-strategy.md](./logos-k-strategy.md) |
| 详细路线图与决策时间表？ | [logos-roadmap.md](./logos-roadmap.md) |
| Logos 64M 验证计划（v1.0/v2.0/v3.0 三阶段）？ | [logos-64m-validation-plan.md](./logos-64m-validation-plan.md) |
| 四种"双时间尺度"实现对比（HRM/Split-GQA/混合）？ | [logos-v1-architecture.md §3](./logos-v1-architecture.md#3-四种双时间尺度实现方案) |
| 五种循环策略延迟-精度 Pareto？ | [logos-v2-architecture.md §4](./logos-v2-architecture.md#4-五策略延迟-精度-pareto-对比) |
| GRAM + Radix Cache 多路径决策？ | [logos-v3-architecture.md §3](./logos-v3-architecture.md#3-radix-cache-多路径集成c2) |

#### SADKO 探索分支（感知 + 记忆 + 知识 + 多模态）

| 问题 | 查 [sadko-whitepaper.md](./sadko-whitepaper.md) 哪一节 |
|---|---|
| 核心设计哲学（压缩即智能 / 异构双脑 / 拓扑对称性）？ | §1 |
| 右脑 ELF / 左脑 AR / 扩散桥梁 / Latent Router 各自怎么实现？ | §2 |
| 为什么选 FSQ 而不是 VQ-VAE？为什么选 Latent MoE 而不是 Token 路由？ | §2.4 决策矩阵 |
| 6+1 个颠覆性创新点？ | §3 |
| 64M 四大证伪实验怎么设计？怎么区分容量问题与架构缺陷？ | §4 |
| Phase 0-8 训练流水线？ | §5 |
| 300M → 1.5B → 7B → 70B 的参数配比与码本扩展？ | §6 |
| 压缩饱和曲线（15:1 → 8:1）与分形码本三级结构？ | §6.2 |
| 可生长认知系统的生物学隐喻与生长原则？ | §7 |
| 技术风险与 Plan B 降级预案？ | §8 |
| 立即执行清单（Fork MiniMind3、核心算子、基线、证伪优先级）？ | §10 |

### 3.2 64M 工程落地（验证计划）

| 问题 | 查 [sadko-64m-validation-plan.md](./sadko-64m-validation-plan.md) 哪一节 |
|---|---|
| MiniMind3 64M 基座的架构与硬约束（768 维 / 8 层 / 8 KV heads）？ | §0 |
| 三阶段各自验证什么？失败判定线？ | §1 总览表 |
| v1.0 左脑改造（Split-GQA / 异构 RoPE / Dual-Path FFN）的设计与验收？ | §2 |
| v2.0 MemPool 压缩-读取管线（192 维 / hot_window 4096）的设计与验收？ | §3 |
| v3.0 ELF-Lite + FSQ + 扩散对齐 + 四大实验的完整方案？ | §4 |
| 64M 专属训练约束（LR 上限 / 强制早停 / 消融隔离）？ | §4.5 |
| 四大实验的决策树（A→B→C→D 顺序与失败处理）？ | §4.6 |
| 阶段间依赖链与"不可逾越"原则？ | §5 |
| v2.0 → v3.0 哪些模块替换、哪些 100% 复用？ | §6 |
| 50 天分日执行清单？ | §7 |

### 3.3 分阶段架构设计（施工图纸）

| 问题 | 查哪份文档哪一节 |
|---|---|
| MiniMind3 64M 基座参数与原始模块结构？ | [v1](./sadko-v1-architecture.md) §2 |
| 锚点层/Cross-Attn 层怎么选？RoPE base 为什么 10k/500k？ | [v1](./sadko-v1-architecture.md) §3 决策表 |
| `SADKOSplitGQA` / `SADKODualPathFFN` 完整代码？ | [v1](./sadko-v1-architecture.md) §5 |
| 如何从 MiniMind3 预训练权重拆分初始化（K/V 投影劈裂）？ | [v1](./sadko-v1-architecture.md) §6 |
| v1.0 验收实验（PPL/熵分离/门控分化/消融）代码？ | [v1](./sadko-v1-architecture.md) §8 |
| MemPool 存储结构与 FIFO 淘汰策略？ | [v2](./sadko-v2-architecture.md) §4.2 |
| 层专属 192→192 读取投影为什么绝不膨胀到 768？ | [v2](./sadko-v2-architecture.md) §2 决策表、§4.3 |
| 压缩触发器（hot_window=4096 滑出压缩）逻辑？ | [v2](./sadko-v2-architecture.md) §4.4 |
| ELF-Lite 双向 Encoder + Flow Matching 实现？ | [v3](./sadko-v3-architecture.md) §4.1 |
| FSQ 量化器（levels/STE/码本冻结）实现？ | [v3](./sadko-v3-architecture.md) §4.2 |
| 扩散对齐训练（Teacher-Student, 0.3KL+0.7CE）？ | [v3](./sadko-v3-architecture.md) §4.6、§5 |
| 四大实验 A/B/C/D 的可执行代码与判定？ | [v3](./sadko-v3-architecture.md) §6 |
| 64M 不可违反的训练约束（LR 上限/早停/消融隔离）？ | [v3](./sadko-v3-architecture.md) §7 |

### 3.4 右脑（ELF）核心设计

| 问题 | 查哪份文档哪一节 |
|---|---|
| 为什么选 Flow Matching + 双向注意力而不是图扩散（GDM）？ | [elf-vs-gdm](./sadko-elf-vs-gdm-review.md) §2 六维裁决 |
| 事实感知三模块（Fact-Aware Flow / 码字锚定 / Fact-Gate）？ | [elf-vs-gdm](./sadko-elf-vs-gdm-review.md) §3.1 |
| 为什么禁止显式图结构损失 $\mathcal{L}_{graph}$？ | [graph-emergence](./sadko-elf-graph-emergence.md) §0 |
| 怎么让图结构"生长"（数据/损失/架构三层策略）？ | [graph-emergence](./sadko-elf-graph-emergence.md) §1 |
| 怎么证明图长出来了（图论指标 + 功能对齐测试）？ | [graph-emergence](./sadko-elf-graph-emergence.md) §2 |
| mRNA/核糖体/蛋白质折叠比喻的精确技术映射？ | [graph-emergence](./sadko-elf-graph-emergence.md) §3 |
| KV 输入与训练数据的设计要求（70/20/10 黄金比例）？ | [graph-emergence](./sadko-elf-graph-emergence.md) §4 |
| 为什么仅靠压缩率不够、必须解压重构？ | [graph-emergence](./sadko-elf-graph-emergence.md) §5 |
| Phase 0 验收阈值、死亡线、逃生舱？ | [phase0-manual](./sadko-elf-phase0-manual.md) §1、§3 |
| Phase 0 监控面板与分阶段干预策略？ | [phase0-manual](./sadko-elf-phase0-manual.md) §2 |
| 新知识如何安全生长（三模式）？旧知识如何防遗忘？ | [lifecycle](./sadko-elf-lifecycle.md) §2、§3 |
| 过时的图结构如何被替代（结构化遗忘）？ | [lifecycle](./sadko-elf-lifecycle.md) §4 |
| 系统存活的六维检查清单（18 项 + 存活条件表）？ | [lifecycle](./sadko-elf-lifecycle.md) §5 |

---

## 4. 维护规范

1. **新增研究记录**：单主题一个文件（`<主题>-<类型>.md`），并在本索引 §2 登记。
2. **讨论记录转正**：当某研究记录沉淀为正式设计提案时，移至 `docs/rfcs/` 并在此标注去向。
3. **合并原则**：多轮讨论总结出现内容重叠时，以最新最全版本为基线去重合并，保留所有独特信息（量化指标、判定阈值、降级预案）。
4. **语言**：中文为主，英文技术术语保留。
