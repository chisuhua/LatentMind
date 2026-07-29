# Logos-Native-64M 三阶段验证计划

> **一句话定位**：基于 MiniMind3 64M 的 Logos 主线验证路线图——v1.0 双时间尺度对比 → v2.0 多种循环策略 → v3.0 GRAM + Radix Cache 多路径决策，10 周单卡 RTX 3090 完成。
> **性质**：Logos 主线 v1.0 之前的关键验证计划（不是阻塞前置，是风险对冲）
> **最后更新**：2026-07-29
> **上游文档**：[logos-whitepaper.md](./logos-whitepaper.md)（架构设计）
> **下游文档**：[logos-v1-architecture.md](./logos-v1-architecture.md) / [logos-v2-architecture.md](./logos-v2-architecture.md) / [logos-v3-architecture.md](./logos-v3-architecture.md)

---

## 0. 与 SADKO 64M 的关键差异

| 维度 | **SADKO 64M** | **Logos 64M**（本计划）|
|------|---------------|--------------------|
| **核心改造** | Split-GQA + 异构 RoPE + Dual-Path FFN（左脑 AR 改造）| **HRM 分层循环 + Split-GQA 借鉴**（H/L 双时间尺度）|
| **记忆范式** | MemPool + FSQ 离散化（**压缩即记忆**）| 不做（这是 SADKO 领域）|
| **循环策略** | 不做（依赖 Cross-Attention 读 SADKO 记忆）| **核心**：K-sweep + 早退 + Radix Cache 多路径 |
| **核心命题** | AR + ELF 异构双脑能否 work？| HRM 分层循环能否在端侧预算内 work？|
| **任务聚焦** | 文本重建 + 码字检索（记忆）| 推理任务 + 决策任务 |

**关键相似**：两者**都借鉴 Split-GQA + 异构 RoPE**——这是 SADKO 和 HRM 在数学上**独立发现**的同一洞察（双时间尺度），Logos 应主动借鉴。详细同构分析见 [logos-whitepaper.md §2.2](./logos-whitepaper.md#22-模块-b分层递归潜空间引擎-750m)。

---

## 1. 三阶段总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  v1.0 (双时间尺度对比)         v2.0 (循环策略对比)         v3.0 (多路径)  │
│  ┌──────────────────┐        ┌──────────────────┐       ┌────────────┐│
│  │ A.1 HRM H/L 风格  │        │ B.1 串行 K 循环    │       │ C.1 GRAM  ││
│  │ A.2 Split-GQA 风格 │───────▶│ B.2 PLT 并行循环  │──────▶│ C.2 Radix ││
│  │ A.3 混合风格      │        │ B.3 Per-Token 早退│       │ C.3 测试时││
│  │ A.4 标准 Transformer│       │ B.4 Radix Cache   │       │   scaling ││
│  └──────────────────┘        │ B.5 层次化推理     │       └────────────┘│
│                              └──────────────────┘                     │
│  验证: 三种双时间尺度       验证: 端侧并行化策略          验证: 多轨迹    │
│  谁最优                     哪种组合最优                 决策端侧化    │
│                                                                         │
│  周期: 2-3 周                周期: 3-4 周                周期: 3-4 周   │
│  风险: 低                   风险: 中                     风险: 中高    │
│  输出: v2.0 的基线架构       输出: 推荐循环策略组合       输出: GRAM+Radix│
└─────────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 核心命题 | 一句话目标 | 失败判定 |
| :--- | :--- | :--- | :--- |
| **v1.0** | 哪种"双时间尺度"实现最优？| 找到 v2.0 的基线架构（A.3 混合应最优）| 三种方案无差异（说明双时间尺度无价值）|
| **v2.0** | 端侧约束下哪种循环策略最优？| K=2 + 早退 + Radix Cache 三件套 | 所有并行方案都不 work |
| **v3.0** | GRAM 多轨迹能否端侧化？| Radix Cache 让 4 路径 K=2 端侧可行 | GRAM 在 64M 训练崩溃 |

---

## 2. v1.0：双时间尺度对比（基座适配）

### 2.1 核心命题

**SADKO 和 HRM-Text 独立发现同一洞察——"双时间尺度"。但实现不同。Logos 应该用哪种？**

### 2.2 四种实现方案

| 实验 | 配置 | 假说 |
|------|------|------|
| **A.1 HRM 风格** | H block + L block 独立，2H×3L=6 串行 | 块级时间尺度分离最强 |
| **A.2 Split-GQA 风格** | 单 block 内 KV heads 分裂（4 Static + 4 Dynamic）| head 级时间尺度分离 |
| **A.3 混合风格** | H/L block + 各自内部 Split-GQA | 双层时间尺度（应最优）|
| **A.4 基线** | 标准 Transformer block | 对照 |

### 2.3 关键产出

- 找出 **A.3 混合风格**应最优的实证
- 提供 v2.0 的基线架构（混合 H/L + Split-GQA）
- 验证 SADKO 的 Split-GQA 在 Logos 端有效

### 2.4 周期与资源

- **周期**：2-3 周
- **硬件**：单卡 RTX 3090 24GB
- **训练数据**：4B tokens（MiniMind3 原始 Pretrain 数据子集）

详细实施：[logos-v1-architecture.md](./logos-v1-architecture.md)

---

## 3. v2.0：多种循环策略对比

### 3.1 核心命题

**端侧 K ≤ 2 硬约束下，哪种循环策略能突破 K=2 限制？**

### 3.2 五种循环策略

| 实验 | 策略 | 端侧延迟 | 适用场景 |
|------|------|---------|---------|
| **B.1 串行 K 循环** | K=1,2,4,8 串行 | 1x, 2x, 4x, 8x | 基线 |
| **B.2 PLT/HLT-PLT 并行** | CLP + G-SWA 移植 | **≈ 1x**（K=8）| 单轨迹推理 |
| **B.3 Per-Token Early Exit** | 动态 K，per-token 早退 | 平均 ≤ 3 | 通用 |
| **B.4 Radix Cache 多路径** | N 路径并行 + 共享前缀 | **≈ 1x**（N 路径）| 多假设决策 |
| **B.5 层次化推理** | 主 + 子并行 | K_main + max(K_sub) | 可分解推理 |

### 3.3 关键决策点

```
B.1 + B.2 结果：
  → 串行 K=2 vs PLT K=8 并行的延迟-精度 Pareto
  → 找到 v1.0 端侧的最优循环策略

B.3 结果：
  → per-token 早退平均循环数
  → 决定 v1.0 默认是否启用早退

B.4 + B.5 结果：
  → Radix Cache vs 层次化的延迟-精度 Pareto
  → 找到 GRAM v3.0 多路径决策的端侧方案
```

### 3.4 周期与资源

- **周期**：3-4 周
- **硬件**：单卡 RTX 3090 24GB
- **训练数据**：与 v1.0 共享

详细实施：[logos-v2-architecture.md](./logos-v2-architecture.md)

> 📌 **RDD-0001 整合**：原 [RDD-0001-k-sweep-experiment.md](../rfcs/RDD-0001-k-sweep-experiment.md) 已**合并到本计划 v2.0 B.1**，保留 RFC 引用。

---

## 4. v3.0：GRAM + Radix Cache 多路径决策

### 4.1 核心命题

**GRAM 多轨迹能否在端侧预算内完成？**

### 4.2 三种 GRAM 实现

| 实验 | 配置 | 端侧可行性 |
|------|------|----------|
| **C.1 GRAM 单轨迹基线** | K=2 + ε 注入（v1.0 直接用）| ✅ 已可行 |
| **C.2 GRAM + Radix Cache** | 4 路径并行 + 共享前缀 | ✅ 延迟 ≈ 1x K=8 |
| **C.3 GRAM + 层次化** | 主推理 + 子推理并行 | ✅ 延迟 ≈ 2x K=2 |

### 4.3 关键决策点

```
C.2 vs C.3 结果：
  → 多假设决策质量（端侧预算内）
  → 决定 v1.5 升级时是否集成 Radix Cache

测试时 scaling 曲线：
  → N 路径 vs K 深度的 Pareto
  → 推荐 v1.5 端侧多轨迹配置
```

### 4.4 周期与资源

- **周期**：3-4 周
- **硬件**：单卡 RTX 3090 24GB
- **训练数据**：决策任务数据集（自构建 + nuScenes 子集）

详细实施：[logos-v3-architecture.md](./logos-v3-architecture.md)

---

## 5. 验收标准

### 5.1 v1.0 验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| A.1 HRM 风格 PPL | 与 A.4 基线持平 | 优于基线 5% |
| A.2 Split-GQA PPL | 与 A.4 基线持平 | 优于基线 5% |
| A.3 混合 PPL | **优于 A.1 和 A.2** | 优于基线 10% |
| **核心结论** | **A.3 是最优** | **A.3 比基线提升 > 10%** |

### 5.2 v2.0 验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| B.1 串行 K=2 vs K=8 精度差 | < 5% | < 2% |
| B.2 PLT K=8 并行延迟 vs 串行 K=2 | < 1.5x | < 1.2x |
| B.3 Per-Token 早退平均 K | ≤ 3 | ≤ 2.5 |
| B.4 Radix Cache 4 路径延迟 | ≈ 1.5x K=2 | ≈ 1.2x K=2 |
| B.5 层次化推理延迟 | ≈ 3x K=1 | ≈ 2.5x K=1 |
| **核心结论** | **B.2 + B.3 + B.4 三件套** | **覆盖所有场景** |

### 5.3 v3.0 验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| GRAM 64M 训练稳定 | ELBO 收敛 | 收敛且无 mode collapse |
| C.2 多路径决策质量 | 比单轨迹提升 ≥ 5% | 提升 ≥ 10% |
| C.3 层次化决策质量 | 比单轨迹提升 ≥ 3% | 提升 ≥ 8% |
| 测试时 scaling 曲线 | N 路径单调增 | N 路径饱和点清晰 |
| **核心结论** | **Radix Cache 是端侧 GRAM 的最优解** | **N=4 路径 K=2 是推荐配置** |

---

## 6. 时间线（10 周单卡 3090）

```
Week 1-3:   v1.0 双时间尺度对比
            ├─ W1: A.1 + A.4 基线
            ├─ W2: A.2 + A.3 实现
            └─ W3: 数据收集 + 验收

Week 4-7:   v2.0 多种循环策略对比
            ├─ W4: B.1 串行 K-sweep
            ├─ W5: B.2 PLT 移植
            ├─ W6: B.3 早退 + B.4 Radix Cache
            └─ W7: B.5 层次化 + Pareto 曲线

Week 8-10:  v3.0 GRAM + Radix Cache
            ├─ W8: C.1 GRAM 基线
            ├─ W9: C.2 Radix Cache + C.3 层次化
            └─ W10: 测试时 scaling + 验收

Week 10+:   交付《Logos 64M 验证报告》
            → 推荐 v1.0 端侧架构 + 循环策略 + GRAM 配置
            → 启动 HRM-Text 1B 集成（用 64M 推荐参数）
```

**与 SADKO 64M 关系**：
- 周 1-3 与 SADKO v1.0 并行（共享 MiniMind3 基座）
- 周 4-10 各自独立（SADKO 走 v2.0/v3.0 记忆/多模态路线）

**与 1B 集成关系**：
- Week 10 出《验证报告》→ 启动 HRM-Text 1B 集成
- 1B 集成**复用** 64M 推荐参数（K 值、循环策略、量化方案）

---

## 7. 资源需求

| 资源 | 数量 | 备注 |
|------|------|------|
| **GPU** | 1× RTX 3090 24GB | 复用 SADKO 64M 训练硬件 |
| **训练时长** | ~10 周 × 40h = 400h | 三阶段合计 |
| **存储** | ~500 GB | checkpoints + logs + 决策数据集 |
| **人力** | 1 人 × 10 周 | 全栈（数据 + 训练 + 评估）|

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| A.3 混合风格不优于 A.1/A.2 | 🟡 低 | 中 | 退回 A.1（最稳健的 HRM 风格）|
| PLT 移植精度损失大 | 🟠 中 | 高 | STARS 谱正则化修复（[stars.md](../references/stars.md)）|
| Radix Cache 实现复杂 | 🟠 中 | 中 | 先用 PyTorch 原生实现，v1.0 再优化 |
| 层次化推理难以端到端训练 | 🟡 低 | 中 | 退回单层推理 + 多路径 |
| GRAM 64M 训练不稳定 | 🟠 中 | 高 | 简化变分目标（不训练 μ, σ）|

---

## 9. 决策传递：64M → 1B

### 9.1 64M 出《验证报告》后

| 64M 结论 | 1B 集成策略 |
|---------|-----------|
| A.3 混合最优 | HRM-Text 1B 加 Split-GQA 微调 |
| B.1 K=2 充足 | 1B 集成默认 K=2 |
| B.3 早退有效 | 1B 集成启用 per-token 早退 |
| B.4 Radix Cache 可行 | 1B v1.5 集成 Radix Cache |
| C.2 N=4 路径最优 | 1B v1.5 默认 N=4 路径 K=2 |

### 9.2 失败回退

| 64M 失败 | 1B 集成处理 |
|---------|-----------|
| A.3 不优于基线 | 1B 集成维持 HRM-Text 原方案 |
| B.1 K=2 严重不足 | 触发 K-sweep（Plan B 升级）|
| Radix Cache 不可行 | v1.5 不集成 Radix Cache，退回 K=2 多轨迹 |
| GRAM 不稳定 | v1.5 不集成 GRAM，退回单轨迹 |

---

## 10. 与 SADKO 64M 的协同

### 10.1 共享资源

| 共享项 | 说明 |
|-------|------|
| **MiniMind3 64M 基座** | 同 |
| **训练硬件** | 同一 RTX 3090（不同时段使用）|
| **训练框架** | MiniMind3 原始 train_pretrain.py |
| **部分工具代码** | KV cache 管理、量化算子 |

### 10.2 不共享的设计

| 不共享 | Logos | SADKO |
|--------|-------|-------|
| **核心改造** | HRM H/L + Split-GQA 借鉴 | AR + Split-GQA + Dual-Path FFN |
| **任务** | 推理 + 决策 | 记忆 + 多模态 |
| **循环策略** | 核心研究（K-sweep、Radix Cache）| 不核心（依赖 Cross-Attention 读取）|

### 10.3 互不依赖原则

**任一条路线 64M 失败都不会阻塞另一条**：
- Logos 64M 失败 → 1B 集成退回 HRM-Text 原方案
- SADKO 64M 失败 → Logos 不依赖 SADKO
- 融合失败 → 双路线独立运行

---

## 11. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-whitepaper.md](./logos-whitepaper.md) | Logos 主线架构白皮书（本文档的父级）|
| [logos-k-strategy.md](./logos-k-strategy.md) | K 值策略与端侧可行性（多种并行化策略详细论证）|
| [logos-roadmap.md](./logos-roadmap.md) | Logos 主线路线图（本文档是路线图的 +0 至 +2.5 月详细展开）|
| [logos-v1-architecture.md](./logos-v1-architecture.md) | v1.0 双时间尺度对比详细施工图 |
| [logos-v2-architecture.md](./logos-v2-architecture.md) | v2.0 多种循环策略详细施工图（含 RDD-0001 整合）|
| [logos-v3-architecture.md](./logos-v3-architecture.md) | v3.0 GRAM + Radix Cache 详细施工图 |
| [sadko-64m-validation-plan.md](./sadko-64m-validation-plan.md) | SADKO 64M 验证计划（并行路线）|
| [docs/rfcs/RDD-0001-k-sweep-experiment.md](../rfcs/RDD-0001-k-sweep-experiment.md) | K-sweep RFC（已合并到 v2.0 B.1）|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | PLT 架构（v2.0 B.2 借鉴）|
| [docs/references/stars.md](../references/stars.md) | 崩溃修复（v2.0 Plan B）|
| [docs/research/sadko-v1-architecture.md](./sadko-v1-architecture.md) | SADKO Split-GQA 详细实现（v1.0 A.2 借鉴）|
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 双轨分工战略 |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos 64M 验证计划）
**版本**：v1.0