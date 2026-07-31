# Logos / SADKO 64M 双轨协调文档

> **一句话定位**：LatentMind 双轨（Logos / SADKO）64M 验证的**协调中枢**——明确共享前置验证、并行验证机制、交叉验证框架与决策矩阵，确保两条路线既独立又可比。
> **性质**：双轨项目的核心协调文档（非单一架构文档）
> **最后更新**：2026-07-29（v1.0 新增）
> **关联**：
> - Logos 64M 验证：[64m-validation-plan.md](./64m-validation-plan.md)
> - SADKO 64M 验证：[sadko-64m-validation-plan.md](../sadko/64m-validation-plan.md)

---

## 0. 为什么需要协调文档

> 📌 **2026-07-29 关键缺口发现**：在 Logos / SADKO 都明确为**全新架构、从零训练**后，两条 64M 验证路线有几个**共享前置假设**未单独验证：
> 1. **MiniMind3 64M Dense 从零训练基线**——两路线都依赖，但没人验证
> 2. **3 层 CNN 感知层**——两路线共用模块
> 3. **评估框架**——若不同路线的 PPL / 延迟测量不一致，对比将无效
> 4. **交叉验证**——双轨在 +2 月末需要统一对比，但方案缺失

本协调文档**填补上述空白**，让两条路线**从共享起点出发、独立验证、统一对比**。

---

## 1. 三阶段总览

```
        Phase 0（共享前置，~5 周）
        ├─ P.0.1 MiniMind3 64M 从零训练基线（必需）
        ├─ P.0.2 3 层 CNN 感知层映射（必需）
        ├─ P.0.3 数据 pipeline + tokenizer（必需）
        ├─ P.0.4 训练基础设施（必需）
        ├─ P.0.5 评估框架（必需）
        └─ P.0.6 端侧并行化基础设施（Logos 专用但关键）
                       ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
   Phase 1A (并行 ~10 周)      Phase 1B (并行 ~10 周)
   Logos 64M 验证               SADKO 64M 验证
   (H/L 分层循环 + 循环策略)    (Split-GQA + MemPool + Hippo)
                       ↓
        Phase 2（交叉验证，~2 周）
        ├─ Logos vs SADKO 在 6 类任务上对比
        ├─ 记忆 / 推理 / 多假设 / 决策 / 端侧 / 多模态
        └─ 出《双轨对比报告》
                       ↓
        Phase 3（决策，1 周）
        ├─ 双轨并行 / 单轨升格 / 退回通用
        └─ 启动 Logos 300M 训练
```

**总周期**：~5 + 10 + 2 + 1 = **~18 周**（约 4.5 个月）

---

## 2. Phase 0：共享前置验证（~5 周）

### 2.1 必要性

> **关键论证**：两路线都以 MiniMind3 64M Dense 为基座改造，**但都没有验证 MiniMind3 标准架构本身能从零训练到合理 PPL**。如果 MiniMind3 基座都训不出合理的 PPL（基线 ~2.7-3.0），那么 Logos / SADKO 的改造实验都不可信。

P.0 失败的连锁反应：
- P.0.1 失败 → 两路线都需重新选择基座 → 延后 +1 月
- P.0.2 失败 → 感知层架构变更 → 重做 Logos / SADKO 起步阶段 → 延后 +2 周
- P.0.5 失败 → 跨路线对比无效 → 决策延后

### 2.2 六项共享前置（P.0.1 - P.0.6）

#### P.0.1 MiniMind3 64M Dense 从零训练基线（🔴 必须）

**目标**：复现 MiniMind3 64M 在 4B tokens 上的 baseline PPL

**步骤**：
1. 复制 MiniMind3 训练脚本（train_pretrain.py）
2. 用 4B tokens 子集（参考 MiniMind3 原始 Pretrain）
3. 训练到 baseline PPL < 3.0（C4 验证集，MiniMind3 论文基线）
4. 报告：训练时长、显存峰值、PPL 曲线

**验收标准**：
| 指标 | 合格线 | 优秀线 |
|------|:---:|:---:|
| C4 PPL | < 3.5 | < 3.0 |
| 训练时长（单卡 3090）| < 4h | < 3h |
| 显存峰值 | < 20GB | < 16GB |
| 无 NaN / 崩溃 | 必须 | — |

**失败处理**：
- PPL > 5.0 → 检查数据 + 学习率
- 训崩 → 加 gradient clipping、检查初始化
- 显存超 → 启用 gradient checkpointing

**依赖**：两路线都依赖（共享）。

---

#### P.0.2 3 层 CNN 感知层映射（🟠 必须）

**目标**：验证图像 patch → 768 维 latent space 的映射可训练

**步骤**：
1. 实现 3 层 CNN（参考 SenseNova NEO-Unify）
2. 训练数据：CIFAR-10 / mini-ImageNet 子集
3. 联合损失：图像分类 + latent 对齐损失（与文本 768 维对齐）
4. 验收：分类准确率 + 对齐损失下降

**验收标准**：
| 指标 | 合格线 | 优秀线 |
|------|:---:|:---:|
| CIFAR-10 准确率 | > 70% | > 80% |
| 对齐损失 | < 0.1（cosine sim > 0.9）| < 0.05 |
| 训练时长 | < 2h | < 1h |

**失败处理**：
- 准确率低 → 检查 CNN 容量（可扩展到 5 层）
- 对齐损失不降 → 加 projection layer

**依赖**：两路线共用感知层（共享）。

---

#### P.0.3 数据 Pipeline + Tokenizer（🟠 必须）

**目标**：验证 MiniMind3 tokenizer + 4B tokens 数据 pipeline 顺畅

**步骤**：
1. 复用 MiniMind3 BPE tokenizer
2. 构建数据 pipeline（多线程 DataLoader）
3. 验证文本 / 图像双模态加载
4. 压力测试：连续训练无内存泄漏

**验收标准**：
- Pipeline 吞吐：> 10 samples/sec（单卡）
- 无 OOM / 无数据重复 / 无 token 错位

**依赖**：两路线共用。

---

#### P.0.4 训练基础设施（🟠 必须）

**目标**：单卡 RTX 3090 / 4060 8GB 上能跑 Logos / SADKO 训练

**步骤**：
1. 验证 BF16 / FP16 训练稳定性
2. 验证 gradient checkpointing（应对 64M 长序列）
3. 验证 AdamW + cosine schedule
4. 验证 checkpoint 加载 / 保存
5. 验证断点恢复（训练中断后能续）

**验收标准**：
- BF16 训练稳定（PPL 不退化）
- 显存峰值 < 24GB（3090）
- 可断点恢复（save → load → 继续）

**依赖**：两路线共用。

---

#### P.0.5 评估框架（🟠 必须）

**目标**：建立两路线**统一可比的评估指标**

**必建指标**：
| 类别 | 指标 | 数据集 | 用途 |
|------|------|--------|------|
| **语言建模** | C4 PPL | C4 验证 | 基线对比 |
| **推理** | GSM8K mini PPL + acc | 200 样本 | 推理能力 |
| **决策** | multi-path accuracy | nuScenes mini | 多假设决策 |
| **端侧** | 单 token 延迟（ms）| profile 工具 | 端侧预算 |
| **显存** | 峰值（MB）| nvidia-smi | 量化前预算 |
| **训练算力** | GPU-hours | 时长统计 | 训练成本 |

**禁止各路线私自修改指标**——任何评估必须用 P.0.5 的统一方法。

**依赖**：两路线共用。

---

#### P.0.6 端侧并行化基础设施（🟡 Logos 专用但关键）

**目标**：验证 Radix Cache / Hierarchical / Per-Token 早退在 64M 可实现

**步骤**：
1. 实现简化版 Radix Cache（vLLM prefix caching 思路）
2. 实现 Per-Token 早退机制
3. 实现 Hierarchical reasoning（主 + 子 block）
4. 在 MiniMind3 64M 上跑通（不是改架构，只是跑通基础设施）

**验收标准**：
- Radix Cache：共享前缀命中率 > 50%
- Per-Token 早退：平均循环数 ≤ 3
- Hierarchical：主 + 子推理可端到端跑通

**失败处理**：
- Radix Cache 实现复杂 → 退回简单前缀缓存
- 早退不收敛 → 调整阈值

**依赖**：Logos 专用（SADKO 不依赖），但失败会严重影响 Logos 端侧可行性。

---

### 2.3 Phase 0 时间线（5 周）

```
Week 0: P.0.1 MiniMind3 64M 训练基线（Day 0-3 验证小数据子集）
        P.0.3 数据 pipeline 准备（并行）
        
Week 1: P.0.1 完整 4B tokens 训练（4 小时单卡）
        P.0.2 CNN 感知层实现 + 训练

Week 2: P.0.4 训练基础设施验证
        P.0.2 验证完成

Week 3: P.0.5 评估框架建立
        P.0.6 端侧并行化基础设施

Week 4: Phase 0 验收报告
        ├─ P.0.1 - P.0.6 全部合格
        └─ Phase 1 启动条件达成
```

---

### 2.4 Phase 0 验收报告模板

```markdown
## Phase 0 验收报告

| 任务 | 状态 | 关键结果 | 影响 |
|------|------|---------|------|
| P.0.1 MiniMind3 基线 | ✅/❌ | PPL = X.X, 训练时长 Xh | 决定基座可行性 |
| P.0.2 CNN 感知层 | ✅/❌ | CIFAR-10 acc = X% | 决定多模态基础 |
| P.0.3 数据 pipeline | ✅/❌ | 吞吐 = X samples/sec | 决定训练效率 |
| P.0.4 训练基础设施 | ✅/❌ | 显存 = XGB | 决定训练稳定性 |
| P.0.5 评估框架 | ✅/❌ | 6 类指标就绪 | 决定可比性 |
| P.0.6 端侧基础设施 | ✅/❌ | Radix 命中率 X% | 决定 Logos 端侧 |

**Phase 0 综合结论**：
- [ ] ✅ 全部通过，可启动 Phase 1
- [ ] ⚠️ 部分失败，需修复后重启
- [ ] ❌ 重大失败，重新设计
```

---

## 3. Phase 1：并行验证（~10 周）

### 3.1 并行启动条件

Phase 0 全部 P.0.1 - P.0.6 验收通过后，**两条路线同时启动**。

### 3.2 Logos 64M（Phase 1A）

详见 [64m-validation-plan.md](./64m-validation-plan.md)：
- v1.0：双时间尺度对比（A.1 HRM / A.2 Split-GQA / A.3 混合 / A.4 基线）
- v2.0：循环策略对比（串行 / PLT / 早退 / Radix / 层次化）
- v3.0：多轨迹并行（不集成 GRAM，独立实现）

### 3.3 SADKO 64M（Phase 1B）

详见 [sadko-64m-validation-plan.md](../sadko/64m-validation-plan.md)：
- v1.0：基座适配（Split-GQA + 异构 RoPE + Dual-Path FFN + CA 骨架）
- v2.0：记忆压缩（Shared MemPool + 192 维 + 压缩触发器）
- v3.0：灵魂注入（Hippo-Lite + FSQ + 扩散对齐 + 四大实验）

### 3.4 硬件时间表（关键协调点）

**单一 RTX 3090 共享调度**：

| 工作日 | 时段 | 任务 |
|--------|------|------|
| 周一 | 全天 | Logos 训练 |
| 周二 | 全天 | Logos 训练 |
| 周三 | 全天 | Logos 训练 |
| 周四 | 全天 | SADKO 训练 |
| 周五 | 全天 | SADKO 训练 |
| 周六 | 全天 | SADKO 训练（周末加班）|
| 周日 | 半天 | 共享评估 / 协调 |

**冲突解决**：
- 若 Logos 评估需要 GPU → 暂停 SADKO 训练
- 若 SADKO 紧急消融需要 → 暂停 Logos 训练
- 优先级：共享评估 > 单路线实验

---

## 4. Phase 2：交叉验证（~2 周，Month +5 末）

### 4.1 目标

**在统一评估框架下，对比 Logos 64M vs SADKO 64M 在多任务上的表现**，输出《双轨对比报告》。

### 4.2 六类对比任务

| 类别 | 任务 | 评估重点 | Logos 预期优势 | SADKO 预期优势 |
|------|------|---------|--------------|--------------|
| **推理** | GSM8K mini | 数学推理 PPL + acc | ✅ 分层循环 | — |
| **代码** | MultiPL-E mini | 代码生成 PPL + acc | ✅ 循环精炼 | — |
| **决策** | nuScenes mini multi-path | 多假设决策 acc | — | ⚠️ 多轨迹 |
| **记忆** | 检索 Recall@64 | 事实召回 | — | ✅ MemPool + FSQ |
| **端侧** | 单 token 延迟 | 端侧预算内精度 | ✅ K=2 + 早退 | — |
| **多模态** | CIFAR-10 + 文本对齐 | 跨模态 PPL | ⚠️ 简单感知 | ✅ 流形压缩 |

### 4.3 评估方法

**统一评估脚本**：
```python
def unified_eval(model_path, task_spec):
    """所有路线用同一脚本评估"""
    model = load_model(model_path)
    metrics = {}
    for task in task_spec:
        metrics[task.name] = evaluate(model, task.dataset, task.metric)
    return metrics
```

**不允许各路线私自评估**——必须用统一脚本，结果才有可比性。

### 4.4 对比矩阵模板

| 任务 | Logos 64M | SADKO 64M | 差距 | Logos 优势场景 | SADKO 优势场景 |
|------|----------|----------|------|--------------|--------------|
| GSM8K mini | PPL=X, acc=Y% | PPL=X', acc=Y'% | Δ | 推理 | — |
| MultiPL-E mini | ... | ... | Δ | 代码 | — |
| 决策 multi-path | ... | ... | Δ | — | 决策 |
| 检索 Recall@64 | ... | ... | Δ | — | 记忆 |
| 单 token 延迟 | X ms | X' ms | Δ | 端侧 | — |
| 多模态 | ... | ... | Δ | 简单 | 流形 |

---

## 5. Phase 3：决策（Month +6 末）

### 5.1 决策矩阵

```
Phase 2 结果 → 决策
├─ Logos 推理强 + SADKO 记忆强 + 融合接口可行
│   → ✅ 双轨并行（首选）
│      → Logos 升格主线（推理 + 决策）
│      → SADKO 升格探索分支（感知 + 记忆 + 多模态）
│      → +6 月启动 Logos 300M 训练，SADKO 1.5B 训练
│
├─ Logos 在多任务胜出（含部分记忆能力）
│   → Logos 升格主线，SADKO 备份
│      → Logos 300M 加重记忆能力（探索 MemPool 集成？）
│      → SADKO 退化，等 Logos 1B 完成后重启
│
├─ SADKO 在多任务胜出（含部分推理能力）
│   → SADKO 升格主线，Logos 备份
│      → SADKO 1.5B 加重推理能力
│      → Logos 退化，等 SADKO 1.5B 后重启
│
└─ 双轨都弱或单方严重失败
    → ⚠️ 退回通用 Transformer，找新方向
```

### 5.2 决策时间表

| 时点 | 决策 | 触发条件 |
|------|------|----------|
| Month +0.5 | 启动 Phase 0 | P.0.1 MiniMind3 基线 |
| Month +1.5 | Phase 0 完成验收 | 6 项全部通过 |
| Month +2 | Phase 1A 启动 | Logos 64M 训练 |
| Month +3 | Phase 1B 启动 | SADKO 64M 训练（与 1A 部分并行）|
| Month +5 | 64M 验证完成 | v1/v2/v3 阶段验收 |
| Month +5.5 | Phase 2 交叉验证 | 双轨对比 |
| Month +6 | Phase 3 决策 | 双轨 / 单轨升格 / 退回 |
| Month +6+ | 启动 300M 训练 | 决策后启动 |

---

## 6. 资源需求汇总

### 6.1 计算资源

| 阶段 | 算力 | 备注 |
|------|------|------|
| Phase 0 | 1× RTX 3090 × 5 周 | 共享 |
| Phase 1A Logos | 1× RTX 3090 × 10 周 | 周一-三优先 |
| Phase 1B SADKO | 1× RTX 3090 × 10 周 | 周四-六优先 |
| Phase 2 | 1× RTX 3090 × 2 周 | 共享 |
| Phase 3 | 0 | 仅决策会议 |

### 6.2 人力

| 阶段 | 人力 |
|------|------|
| Phase 0 | 0.5 人 × 5 周（数据 + 评估基础设施）|
| Phase 1A Logos | 1 人 × 10 周 |
| Phase 1B SADKO | 1 人 × 10 周（并行）|
| Phase 2 评估 | 0.5 人 × 2 周 |

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| **P.0.1 MiniMind3 基线失败** | 🟡 低 | 🔴 高 | 先试 0.5B tokens 子集，调优后跑完整基线 |
| **P.0.2 CNN 感知层失败** | 🟠 中 | 🟠 中高 | 调整层数 / 加投影层 |
| **Phase 1 单路线严重超时** | 🟠 中 | 🟡 中 | 暂停对路线，让领先路线先完成 |
| **Phase 2 评估发现不可比** | 🟡 低 | 🔴 高 | P.0.5 评估框架必须严格执行，禁用私自评估 |
| **Phase 3 决策死锁** | 🟡 低 | 🟠 中高 | 有 4 个备选决策，总能前进 |

---

## 8. 协调检查清单

每周末执行：

- [ ] 本周 Logos 进展
- [ ] 本周 SADKO 进展
- [ ] Phase 0 / 1 / 2 状态
- [ ] 硬件使用情况
- [ ] 待解决冲突

每月底执行：

- [ ] 月度 Phase 评估
- [ ] 决策矩阵更新
- [ ] 下月计划

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [64m-validation-plan.md §0](./64m-validation-plan.md) | Logos 64M 验证（引用本协调文档）|
| [sadko-64m-validation-plan.md §0](../sadko/64m-validation-plan.md) | SADKO 64M 验证（引用本协调文档）|
| [whitepaper.md §2](./whitepaper.md) | Logos 主线架构 |
| [sadko-whitepaper.md §2](../sadko/whitepaper.md) | SADKO 主线架构 |
| [roadmap.md §5](./roadmap.md) | Logos 路线图（含双轨协同）|
| [AGENTS.md §7](../../AGENTS.md#7-研究路线分工双轨制--2026-07-29-战略决策) | 双轨分工战略 |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos / SADKO 双轨协调）
**版本**：v1.0