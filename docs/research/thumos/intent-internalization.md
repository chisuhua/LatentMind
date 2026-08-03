# Thumos 意图与触发内化（Intent + Trigger Internalization）

> **方向**：方向 1 / 3（意图与触发 / 编排原语 / 离线在线动力学）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[thumos/README.md §1](../README.md#1-三个并行研究方向)
> **V4 来源**：议题 1（顶层架构：Intent Session）、议题 10（双层动态触发）、概念 13（AR Uncertainty 盲区）、概念 14（四种 Join 模式 - L0/L1 子集）、❓13、❓18

---

## 0. 核心问题

L0 意图解析与 L1 AR 内部信号的协同能否**内化为模型 forward pass 内的反馈环路**，替代外部分类器 + 外部不确定性监控？

**具体子问题**：

1. Intent 解析能否作为模型自身的循环模块（前向计算图），而非外部分类器？
2. 双层触发（L0 意图层 + L1 AR 信号）的协同回路能否在单次 forward pass 内闭环？
3. 意图漂移检测能否内化为 Hidden State 的统计指标（而非外部状态机）？
4. 内化 Intent + 触发是否在端到端任务上**不劣于**外部 Intent Session + 外部监控（[thumos/README.md §0.1 INV-3 兜底](../README.md#03-与-hydraforge--agenticllama-的边界)）？

---

## 1. 研究目标

定义 Thumos 内化 Intent + 触发的最优架构：

- (a) Intent 解析作为 forward pass 内的轻量子模块（< 50M 参数）
- (b) 双层触发内部反馈回路（< 50ms 反馈延迟）
- (c) 意图漂移内部检测（基于 Hidden State 统计）
- (d) 与 HydraForge 的降级路径（INV-3：内化失败 → 外化接管 100% 成功）

---

## 2. 最小验证单元

`InternalIntentTrigger` 模块：

- (a) 输入 = 当前 step 的 Hidden State + AR Uncertainty 信号
- (b) 输出 = 意图向量（赫尔墨斯契约 H1 Intent Snapshot）+ 触发决策（是否发起检索 / 触发哪类 Join 模式）
- (c) 支持内化失败时静默降级到"无意图无触发"状态

**下游消费**：Hippo 检索 API（[I3 Retrieval API](../../research/hippo/README.md#22-接口契约五元组骨架)）+ Logos Nano-WM Gate（[方向 2](./fork-join-internalization.md)）

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| Intent 解析准确率 | L0 意图分类与外部 ground truth 一致 | ≥ 0.85 |
| 双层触发反馈延迟 | L0 触发 → L1 信号 → 决策的总耗时 | < 50ms |
| 意图漂移检测 F1 | 基于 Hidden State 统计的漂移检测 | ≥ 0.7 |
| 端到端任务 delta | 内化 vs 外部 Intent Session | 内化 ≥ 外部 - ε |
| INV-3 降级成功率 | 内化失败时 HydraForge 接管成功率 | 100% |

---

## 4. 前置依赖

- [`../README.md`](../README.md)（Thumos 线路索引，含赫尔墨斯契约）
- [`../logos/`](../logos/)（Logos 主线 Nano-WM 与 H/L 循环——本方向消费其 Hidden State）
- [`../hippo/README.md`](../hippo/README.md)（Hippo I3 检索 API——L1 信号源之一）
- V4 议题 10 + 概念 13/14（双层触发 + AR Uncertainty 盲区 + 四种 Join 模式 - L0/L1 子集）

---

## 5. 待填充内容（实施时）

- [ ] Intent 解析作为 forward pass 内轻量子模块（< 50M 参数）的网络结构
- [ ] 双层触发反馈回路的时序控制（如何让 L0 与 L1 在单次 forward pass 内闭环）
- [ ] 意图漂移的内部检测指标（基于 Hidden State 统计）
- [ ] 内化失败检测器（用于触发 INV-3 降级到 HydraForge）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约 H1/H3/INV-3）

---

## 6. 鲁棒性评分（待填）

[待实施后按 thumos/README.md §3.3 评分]

---

## 7. 64M 阶段 Riskiest Bet 实验

**内化 vs 外部 vs 无编排对比**（必须含完全离线模式）：

| 组别 | 描述 | 评估指标 |
|------|------|---------|
| (i) **内化 Intent + 触发** | 模型自己解析 Intent，forward pass 内 L0↔L1 闭环 | 任务 acc / 反馈延迟 |
| (ii) **外部 Intent Session + 监控** | 模拟 HydraForge：外部 Intent 解析 + Uncertainty 监控 | 任务 acc / 反馈延迟 |
| (iii) **无编排基线** | 无 Intent 解析、无触发反馈 | 任务 acc |

**判定**：
- (i)≈(ii)≫(iii) ⇒ 内化不劣于外部（验证通过）
- (i)≫(ii) ⇒ 内化胜利（内化假设强成立）
- (i)<(ii) ⇒ 内化假设失败（标记"已证伪"，依赖 INV-3 兜底）

---

## 8. 与 V4 议题的对应

| V4 来源 | 映射 |
|---------|------|
| 议题 1（顶层架构：Intent Session） | 本方向核心 |
| 议题 10（双层动态触发：L0 意图层 + L1 AR 信号） | 本方向核心 |
| 概念 13（AR Uncertainty 盲区） | L1 信号源 |
| 概念 14（四种 Join 模式 - L0/L1 子集） | 触发决策的输出空间 |
| ❓13（Intent Session 架构与训练） | 本方向 §1 锁定 |
| ❓18（意图漂移检测阈值策略） | 本方向 §3 关键指标 |

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。

**最后更新**：2026-07-31