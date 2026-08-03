# Thumos 编排原语内化（Fork/Join Internalization）

> **方向**：方向 2 / 3（意图与触发 / 编排原语 / 离线在线动力学）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[thumos/README.md §1](../README.md#1-三个并行研究方向)
> **V4 来源**：议题 2（Fork/Join 协作机制）、议题 7（Nano-WM）、议题 9（分层注入）、概念 8（Nano-WM FFN 并行 + 门控）、概念 11（事实 vs 决策分开注入）、概念 14（四种 Join 模式 - Soft/Hard/Feedback/Fork）、❓1、❓2、❓5、❓14、❓17

---

## 0. 核心问题

Soft / Hard / Feedback / Fork 四种 Join 模式能否作为**模型架构机制**（前向计算图 + Gate + Control Token + 内部反馈回路），而非服务层调度决策？

**具体子问题**：

1. Soft Join（Cross-Attention / Nano-WM 门控）的内化已经在 Logos 主线完成——是否需要 Thumos 进一步优化（如逐 Token Gate 信号路由）？
2. Hard Join（Control Token 插入）的内化如何处理不可微性（REINFORCE / Gumbel-Softmax）？
3. Feedback Join（Uncertainty 反向触发）能否在 forward pass 内实现"内部回路"（不依赖外部异步调用）？
4. Fork Point（Prefill 后分流）能否作为模型的"并行启动机制"（共享 KV Cache + 多 Head 异步流）？

---

## 1. 研究目标

定义 Thumos 内化四种 Join 模式的完整架构：

- (a) **Soft Join** → 复用 Logos Nano-WM Gate（已是架构机制），Thumos 优化 Gate 信号源路由
- (b) **Hard Join** → 模型词表扩展 + Control Token 嵌入（需内化验证）
- (c) **Feedback Join** → 模型 forward pass 内反馈回路（需内化验证）
- (d) **Fork Point** → Prefill 后 KV 分流机制（需内化验证）

**赫尔墨斯契约 H2**（Join Mode Vector）必须能精确表达当前 step 选择的 Join 模式（4 维 one-hot）+ Gate 值（标量 ∈ [0,1]）。

---

## 2. 最小验证单元

`InternalForkJoin` 模块：

- (a) 输入 = 当前 step Hidden State + L0 Intent（[方向 1](./intent-internalization.md)）+ L1 Uncertainty
- (b) 输出 = (Join Mode one-hot, Gate scalar, [Optional] Control Token)
- (c) 与下游 Nano-WM Gate / Hippo 检索 API / Logos KV 分流机制的标准对接

**下游消费**：

| Join 模式 | 消费方 |
|-----------|--------|
| Soft Join | Logos Nano-WM 并行 FFN（方向 1 已锁定 Gate 信号）|
| Hard Join | 模型词表（需新增 Control Token 嵌入）|
| Feedback Join | 内部反馈回路（自循环，不依赖外部异步）|
| Fork Point | Logos Prefill 后 KV 分流（与 Logos KV Cache 共享）|

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| Soft Join Gate 有效性 | 注入真知识 vs 随机 delta | ≥ +15% 任务 acc |
| Hard Join 触发准确率 | Control Token 触发后行为变化符合预期 | ≥ 0.9 |
| Feedback Join 反馈延迟 | Uncertainty 触发 → 增量检索 → 结果注入 | < 100ms |
| Fork Point 并行效率 | Prefill 后 KV 分流的并行加速比 | ≥ 1.5x |
| H2 接口契约满足度 | Join Mode Vector 形态与 HydraForge 对齐 | 100% |
| 四模式协同一致性 | 多模式联合使用的内部一致性 | 无冲突 |
| INV-2 退化保证 | Gate=0 时退化为原始 Logos 行为 | 100% |

---

## 4. 前置依赖

- [`../README.md`](../README.md)（Thumos 赫尔墨斯契约 H2 + INV-2）
- [`../logos/whitepaper.md`](../logos/whitepaper.md)（Logos Nano-WM 与多轨迹并行——Soft Join 内化基础）
- [`../sadko/whitepaper.md`](../sadko/whitepaper.md)（SADKO 扩散对齐——Hard Join 可借鉴不可微处理）
- V4 议题 2 / 7 / 9（Fork/Join / Nano-WM / 分层注入）
- V4 概念 8 / 11 / 14（Nano-WM / 事实 vs 决策 / 四种 Join 模式）
- [方向 1：Intent + Trigger](./intent-internalization.md)——Fork/Join 消费的 Intent 来源

---

## 5. 待填充内容（实施时）

- [ ] 四种 Join 模式各自的内化网络结构
- [ ] Hard Join 不可微处理（REINFORCE / Gumbel-Softmax 在端侧的实施）
- [ ] Control Token 词表设计（新增 vs 复用）+ 嵌入训练方案
- [ ] Feedback Join 的内部回路设计（不依赖外部异步调用）
- [ ] Fork Point 的 Prefill 后 KV 分流机制（与 Logos KV Cache 共享）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（四模式分别 / 多模式协同 / 消融 / 超参扫描 / 接口契约 H2/INV-2）

---

## 6. 鲁棒性评分（待填）

[待实施后按 thumos/README.md §3.3 评分]

---

## 7. 64M 阶段 Riskiest Bet 实验

**四模式分别内化对比**：

| 模式 | 对照组 | 实验组 |
|------|--------|--------|
| **Soft Join** | 外部 Nano-WM Gate | 内化 Gate（信号源自方向 1 L1） |
| **Hard Join** | 无 Hard Join（仅 Soft Join） | 内化 Control Token 触发 |
| **Feedback Join** | 外部异步触发 | 内化 forward pass 反馈 |
| **Fork Point** | 串行 Prefill | 内化 Prefill 后 KV 分流 |

**判定**：
- 每模式：内化 ≥ 外部 - ε（验证通过）
- 四模式协同：内部一致性无冲突 + Gate=0 严格退化（INV-2 满足）

---

## 8. 与 V4 议题的对应

| V4 来源 | 映射 |
|---------|------|
| 议题 2（Fork/Join 协作机制） | 本方向核心 |
| 议题 7（Nano-WM 范式：FFN 并行 + 门控） | Soft Join 内化基础 |
| 议题 9（分层注入策略：事实 vs 决策） | Soft Join（事实）vs Hard Join（决策）的路由依据 |
| 概念 8（Nano-WM FFN 并行 + 门控） | Soft Join 实现 |
| 概念 11（事实 vs 决策分开注入） | Soft vs Hard 路由规则 |
| 概念 14（四种 Join 模式） | 本方向 §1 全部产出 |
| ❓1（Nano-WM 每层 vs 选择性层） | Soft Join 实施 |
| ❓2（Gate 信号来源） | 与方向 1 的 L1 信号协同 |
| ❓5（注入路径中间地带） | Soft vs Hard 路由边界 |
| ❓14（Control Token 词表设计） | Hard Join 实施 |
| ❓17（三路 Session 模型共享底层参数） | Fork Point 与 Logos KV 共享 |

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。

**最后更新**：2026-07-31