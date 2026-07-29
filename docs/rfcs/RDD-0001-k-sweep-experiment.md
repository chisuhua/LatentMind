# RFC: HRM-Text 循环深度 K-sweep 前置实验（RDD-0001）

> **状态**：📝 **Plan B 草案**（v1.1 — 2026-07-29 **降级**）
> **作者**：来自工作流（LoopCoder-v2 论文冲击波分析）
> **目标规模**：MiniMind3 64M（快速消融）→ HRM-Text 1B（条件性验证）
> **目标场景**：作为 **Logos 主线集成的 Plan B 预案**——仅在 K=8 集成失败时启动
> **触发条件**（满足任一即启动）：
>   - K=8 推理时延超过 ChipForge APU 预算
>   - K=8 出现 gain-then-collapse 现象
>   - 端侧 K 必须 ≤ 2 时
> **与项目的关系**：**不再是主线前置阻塞**，而是集成失败后的快速消融方案
> **关联文档**：
> - 上游论证：[docs/references/loopcoder-v2.md](../references/loopcoder-v2.md)
> - 反例：[docs/references/huginn.md](../references/huginn.md)
> - 修复方案：[docs/references/stars.md](../references/stars.md)
> - 动态 K 证据：[docs/references/per-token-convergence.md](../references/per-token-convergence.md)
> - Logos K 策略：[../research/logos-k-strategy.md](../research/logos-k-strategy.md)
> - HRM-Text 主线：[docs/references/hrm-text.md](../references/hrm-text.md)

> ⚠️ **本 RFC 是实验设计提案，未实现也未验证**。所有 K 值、指标阈值、资源估算均基于理论推演 + 外部论文数据，正式实验需重新校准。

---

## 1. 摘要

### 1.1 一句话总结（Plan B 版本）

**作为 Logos 集成失败的 Plan B：在 MiniMind3 64M 上跑 K ∈ {2, 4, 8} 快速消融（<10h），找出在不超端侧时延预算前提下的最优 K 值**。

### 1.2 实验范围收缩（vs 原方案）

| 维度 | 原阻塞方案 | Plan B 方案 |
|------|----------|-----------|
| K 值数 | 6（{1, 2, 4, 6, 8, 12}）| 3（{2, 4, 8}）|
| 子实验数 | 4（A/B/C/D）| 2（仅 A + B）|
| 数据集数 | 5（GSM8K + HumanEval+ + ARC-AGI + WikiText + MultiPL-E）| 1（GSM8K，最关键）|
| 训练时长 | ~54h | <10h |
| 时序 | 主线前置（+0 月启动）| 集成失败后启动 |

> 📌 **核心降级理由**：LoopCoder-v2 已证明循环架构方向正确（K=2 > K=1），K-sweep 不再需要"验证架构方向"，只需"找最优点"。范围收缩 5×。

### 1.2 关键问题

HRM-Text 论文采用 K=4-8（2H×3L）层次化循环。LoopCoder-v2 实证发现 PLT 架构在 K=3 崩溃（7B SWE-bench 64.4 → 27.6），但 Huginn 反例显示 3.5B 无 CLP 循环可 K=50 仍 work。**HRM-Text 的 K=4-8 假设是否成立，必须独立验证**。

### 1.3 决策点

实验结果将决定主线 v1.0 的 K 值选择：

| 实验结果 | 主线 v1.0 K 决策 |
|---------|-----------------|
| K=8 仍 work，层次化免疫崩溃 | 维持 K=8（论文原方案）|
| K=4 后开始退化 | 改 K=2-4 + per-token early exit |
| K=2 后仍退化 | 启用 STARS 谱半径正则化 |
| 任意 K 都崩 | 切换到 Huginn 风格（无 CLP 顺序循环）|

---

## 2. 背景与动机

### 2.1 LoopCoder-v2 的冲击

详见 [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md)。核心数字：

| 架构 | 7B SWE-bench 表现 |
|------|-------------------|
| R=1 (基线) | 43.0% |
| **R=2** | **64.4%**（+50%）|
| R=3 | 27.6%（**比基线差**）|
| R=4 | 22.4% |

**3 个崩溃机制**（论文诊断）：
1. 表示空间坍缩（effective rank 下降）
2. 振荡而非收敛（$\cos\theta^{(r)} < 0$）
3. 固定位置偏移税（PLT 的 CLP 偏移特有）

### 2.2 HRM-Text 是否免疫？

| 维度 | PLT（崩溃）| HRM-Text（待验证）|
|------|-----------|-----------------|
| Loop 结构 | 扁平 + 共享参数 | 层次化 H/L 不同参数 |
| 信息流 | CLP 右移一位 | 加法注入 $z_L + z_H$ |
| 是否有"位置税" | ⚠️ 有（CLP 偏移）| ✅ 无 |
| 是否有"慢变量锚点" | ❌ 无 | ✅ H 模块（慢速）|

**理论推断**：HRM-Text 的层次化结构**可能**免疫 PLT 的崩溃（因为 H 模块提供慢变量锚点，防止 L 模块进入低维坍缩空间）。**但这只是理论推断，必须实验验证**。

### 2.3 端侧时延约束

K=8 意味着推理时延 × 8 倍。在 ChipForge APU 端侧：
- 假设单次前向 50ms（HRM-Text 1B FP16）
- K=8 推理 400ms（不可接受）
- K=2 推理 100ms（可接受）
- K=1 推理 50ms（理想）

**端侧 K ≤ 2 是硬约束**。如果实验显示 K>2 才能达到精度门槛，必须：
1. 启用 per-token early exit（90% token K=2 退出）
2. 启用 STARS 谱半径正则化（提升 K=2 时的稳定性）
3. 移植 PLT 的 CLP 并行化（推理延迟 × K 降为 × 1）

---

## 3. 实验设计

### 3.1 总体结构

```
实验 A：基线 K-sweep（必备）
  → 在 MiniMind3 64M + 标准循环 block 上跑 K ∈ {1, 2, 4, 6, 8, 12}
  → 输出 K-精度曲线 + per-token 收敛步数分布

实验 B：层次化 vs 扁平化（核心）
  → 2H×3L=8（HRM-Text 风格） vs flat 8（PLT 风格）对照
  → 验证层次结构是否免疫崩溃

实验 C：动态 K vs 固定 K（次要）
  → per-token early exit vs 固定 K
  → 验证 per-token 收敛论文的发现

实验 D：STARS 谱半径正则化（修复）
  → 若 A/B 发现 K=6+ 崩溃，启用 STARS 验证修复
  → 测量性能恢复幅度
```

### 3.2 变量与对照

| 实验 | 自变量 | 因变量 | 对照 |
|------|--------|--------|------|
| **A** | K ∈ {1, 2, 4, 6, 8, 12} | 验证集精度 + PPL | 标准循环 Transformer（无层次化）|
| **B** | 循环结构（层次 vs 扁平）| K-精度曲线 | 同样的"有效深度"（如 2H×3L vs flat 8）|
| **C** | K 调度（固定 vs 动态）| 平均 K + 精度 | 固定 K=8 vs per-token early exit |
| **D** | STARS 正则（启用 vs 不启用）| K=8 性能恢复幅度 | 不启用 STARS 的 K=8 |

### 3.3 数据集

| 数据集 | 任务类型 | 用途 | 优先级 |
|--------|---------|------|:---:|
| GSM8K | 数学推理 | 推理任务 | 🔴 核心 |
| HumanEval+ | 代码生成 | 推理任务 | 🔴 核心 |
| ARC-AGI-1 | 抽象推理 | 推理任务 | 🟡 重要 |
| WikiText-103 | 语言建模 | PPL 监控 | 🟡 重要 |
| MultiPL-E | 多语言代码 | 泛化性 | 🟢 加分 |

### 3.4 评估指标

| 维度 | 指标 | 目标 |
|------|------|------|
| **精度** | 任务准确率 | K=8 ≥ K=4 ≥ K=2 ≥ K=1 |
| **稳定性** | PPL 跨 K 的标准差 | < 10% 相对变化 |
| **收敛性** | effective rank 跨 K 的变化 | 单调不减 |
| **效率** | 平均循环数（动态 K）| ≤ 4（vs 固定 8）|
| **时延** | 单 token 推理时间 | K=2 ≤ 100ms |

### 3.5 训练约束

| 维度 | 配置 | 备注 |
|------|------|------|
| 模型 | MiniMind3 64M（768 dim / 8 层 / 8 KV heads）| 见 [docs/research/sadko-v1-architecture.md §2](../research/sadko-v1-architecture.md) |
| 数据 | 4B tokens（MiniMind3 原始 Pretrain）| 与原论文对齐 |
| 训练 tokens | 2 epochs | 避免过拟合 |
| 优化器 | AdamW（β₁=0.9, β₂=0.95）| 与 SADKO 64M 对齐 |
| 学习率 | 1e-4（warmup 2000 步）| SADKO 64M 上限 |
| 硬件 | 单卡 RTX 3090 24GB | SADKO 约束 |
| 训练时长 | ~6h/run × 6 K 值 + 3 对照 = 9 runs × 6h ≈ **54h** | 不含消融 |

---

## 4. 验收标准

### 4.1 实验 A：基线 K-sweep

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| K=2 相对 K=1 精度提升 | ≥ +5% | ≥ +10% |
| K=4 相对 K=2 精度提升 | ≥ +2% | ≥ +5% |
| K=8 相对 K=4 精度提升 | ≥ +0% | ≥ +3% |
| K=12 相对 K=8 精度提升 | ≥ -5% | ≥ +0% |
| **核心结论** | **K=8 仍 work 或 K=6+ 不崩溃** | **K=8 是单调不减的拐点** |

### 4.2 实验 B：层次化 vs 扁平化

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| 2H×3L=8 vs flat 8 在 K=8 时的精度差异 | ≤ 5% | 层次化 ≥ 扁平化 |
| 层次化在 K=12 时是否仍稳定 | 崩溃阈值 ≥ 12 | 崩溃阈值 ≥ 16 |
| **核心结论** | **层次化免疫扁平化崩溃** | **层次化 K=12 仍 work** |

### 4.3 实验 C：动态 K

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| 动态 K 平均循环数 | ≤ 4 | ≤ 3 |
| 动态 K 精度 vs 固定 K=8 | ≥ K=8 - 2% | ≥ K=8 |
| 动态 K 时延降低 | ≥ 30% | ≥ 50% |
| **核心结论** | **动态 K 显著降低时延且不损精度** | **动态 K 精度超越固定 K=8** |

### 4.4 实验 D：STARS 修复（条件性）

**触发条件**：实验 A/B 发现 K=6+ 性能衰退 > 10%。

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| STARS 启用后 K=8 性能恢复 | 恢复 ≥ 50% 衰退量 | 恢复 ≥ 80% 衰退量 |
| STARS 训练开销 | ≤ +20% | ≤ +10% |
| **核心结论** | **STARS 是 K=8 退化的有效修复** | **STARS 让 K=8 接近 K=4 性能** |

---

## 5. 必做消融实验

按优先级排序：

1. **A.1 K-sweep 全量**：K ∈ {1, 2, 4, 6, 8, 12} on GSM8K + HumanEval+
2. **B.1 层次 vs 扁平**：2H×3L=8 vs flat 8 on GSM8K
3. **B.2 层次化 K-sweep**：2H×1L=2, 2H×2L=4, 2H×3L=6, 2H×4L=8 on GSM8K
4. **C.1 per-token 收敛监控**：记录每 token 收敛步数分布
5. **C.2 动态 K vs 固定 K**：early exit threshold 网格搜索
6. **A.2 PPL 监控**：K-sweep on WikiText-103 验证 language modeling
7. **D.1 STARS Jacobian 谱半径监控**：触发后立即执行
8. **D.2 STARS γ sweep**：$\gamma \in \{0.85, 0.90, 0.95, 0.99\}$
9. **B.3 effective rank 监控**：每步循环的 effective rank
10. **A.3 跨任务泛化**：ARC-AGI-1 + MultiPL-E

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|:---:|:---:|---------|
| MiniMind3 64M 训练崩溃 | 中 | 高 | 加 gradient clipping（虽然 HRM-Text 论文不要，但 MiniMind3 可能需要）|
| 6 K 值 × 9 runs 训练时长超预算（54h）| 中 | 中 | 用更小数据子集（1B tokens），延长到 4 epochs |
| per-token 收敛监控实现复杂 | 中 | 低 | 先做 K-sweep，per-token 监控作为后续工作 |
| STARS 谱半径正则化实现复杂 | 中 | 中 | 先做 A/B，STARS 仅在 K>6 崩溃时启用 |
| Huginn 风格循环需重构代码 | 低 | 高 | 仅在所有其他方案失败时启用 |

---

## 7. 实施路径

### 7.1 阶段一：基线 K-sweep（Week 1-2）

```python
# 伪代码
for K in [1, 2, 4, 6, 8, 12]:
    model = LoopedTransformer(depth=8, K=K, share_params=True)
    trainer = Trainer(model, data=gsm8k, lr=1e-4, epochs=2)
    trainer.fit()
    metrics = evaluate(model, test_sets)
    log(K=K, **metrics)
```

### 7.2 阶段二：层次化对比（Week 3）

```python
# 2H×3L 风格
model = HRMTextStyle(dim=768, H_layers=8, L_layers=8, H_cycles=2, L_cycles=3)
# 总循环数 = 2 × 3 = 6 + 2 H updates = 8
```

### 7.3 阶段三：动态 K（Week 4）

```python
def forward_with_early_exit(x, max_K=8, epsilon=1e-4):
    h = init(x)
    for t in range(max_K):
        h_new = block(h, x)
        delta = (h_new - h).norm(dim=-1)
        exit_mask = delta < epsilon
        if exit_mask.all(): break
        h = h_new
    return h
```

### 7.4 阶段四：STARS 修复（条件性，Week 5+）

```python
def stars_loss(model_output, target, jacobian_rho, gamma=0.95):
    ce_loss = cross_entropy(model_output, target)
    spectral_penalty = max(0, jacobian_rho - gamma)
    return ce_loss + 0.1 * spectral_penalty
```

---

## 8. 与 LatentMind 主线 v1.0 的关系

### 8.1 时序约束

```
Week 0-5: RDD-0001 K-sweep 实验
Week 6-7: 决策 HRM-Text K 值 + 是否启用 STARS / early exit
Week 8-9: HRM-Text 权重获取 + 感知层集成
Week 10+: 主线 v1.0 训练（用 K-sweep 决策的 K 值）
```

### 8.2 决策传递

| K-sweep 结论 | v1.0 K 决策 | v1.0 实施动作 |
|-------------|------------|---------------|
| K=8 仍 work | K=8 | 维持 HRM-Text 原方案 |
| K=4 后崩溃 | K=2 + early exit | 改 K=2 + per-token early exit |
| K=2 仍崩 | K=2 + STARS | 启用 STARS 谱半径正则化 |
| 任意 K 崩 | K=2 + Huginn 风格 | 改用顺序循环（无层次化、无 CLP）|

### 8.3 与 [../research/sadko-64m-validation-plan.md](../research/sadko-64m-validation-plan.md) 的关系

两条路线的前置实验**并行**：
- **主线前置**：RDD-0001 K-sweep（验证 K=4-8 假设）
- **SADKO 前置**：sadko-64m-validation-plan.md 64M 四大机制实验

两条路线均完成小规模验证后（Month 2-3），才能决定融合策略（见 [AGENTS.md §7.5](../../AGENTS.md#75-决策时间表)）。

---

## 9. 资源需求

| 资源 | 数量 | 备注 |
|------|------|------|
| GPU | 1× RTX 3090 24GB | 复用 SADKO 64M 训练硬件 |
| 训练时长 | ~54h（实验 A/B）| 不含实验 C/D |
| 存储 | ~500 GB | checkpoints + logs |
| 人力 | 1 人 × 5 周 | 全栈（数据 + 训练 + 评估）|

---

## 10. 决策检查表

完成实验后，逐项回答：

- [ ] K=8 相对 K=4 精度变化？单调性如何？
- [ ] K=12 相对 K=8 精度变化？是否崩溃？
- [ ] 层次化（2H×3L）vs 扁平 8 在 K=8 时差异？
- [ ] per-token 收敛步数分布？是否 90% token ≤ 6 步？
- [ ] 动态 K（early exit）平均循环数？精度损失？
- [ ] STARS 谱半径正则化是否恢复 K=8 性能？

**最终决策**（填入 v1.0 K 值 + 配套机制）：

```
v1.0 K = _____ (候选：1, 2, 4, 8)
v1.0 early exit = _____ (是 / 否)
v1.0 STARS = _____ (启用 / 不启用)
v1.0 循环结构 = _____ (层次化 / 扁平 / Huginn 风格)
```

---

## 11. 附录：相关参考

| 文档 | 作用 |
|------|------|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | K-sweep 实验的根本动机 |
| [docs/references/huginn.md](../references/huginn.md) | 反例：循环可 K=50 |
| [docs/references/stars.md](../references/stars.md) | 崩溃的 LayerNorm 根因 + Jacobian 修复 |
| [docs/references/per-token-convergence.md](../references/per-token-convergence.md) | 90% token 6 步收敛——动态 K 证据 |
| [docs/references/hrm-text.md](../references/hrm-text.md) | 主线 backbone 完整笔记 |
| [docs/references/rrm-survey.md](../references/rrm-survey.md) | RRM-B 谱系全景 |
| [docs/research/sadko-64m-validation-plan.md](../research/sadko-64m-validation-plan.md) | SADKO 路线的前置实验（并行）|
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 双轨分工战略 |

---

**最后更新**：2026-07-29
**草案版本**：v1
