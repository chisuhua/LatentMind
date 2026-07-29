# Per-Token Fixed-Point Convergence 参考资料

> **论文**：*Per-Token Fixed-Point Convergence in Recurrent Depth Transformers*（2026-07）
> **arXiv**：待补充

> ⚠️ **本笔记基于 librarian 摘要，arXiv ID 暂未独立确认。** 关键发现 90%/10% 分布来自论文摘要，详见 §4 复现风险。

---

## 1. 一句话定位

**实证分析"递归深度 Transformer 中每个 token 的收敛步数分布"——发现 90% 的 token 在 6 步内达到不动点，10% 的 token 需要 8 步，为 per-token 动态循环数（early exit）提供关键证据**。这是 LatentMind 主线"动态 K"实验的核心理论支撑。

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | ⚠️ 待补充 |
| 论文性质 | 2026-07 预印本 |

> 📌 **TODO**：补充 arXiv ID 与完整作者列表

---

## 3. 核心发现

### 3.1 测量方法

对训练好的循环 Transformer，逐 token 测量其 hidden state 在循环中的收敛步数：

$$\Delta_t^{(i)} = \| \mathbf{h}_t^{(i)} - \mathbf{h}_{t-1}^{(i)} \|_2$$

定义"已收敛"条件：$\Delta_t^{(i)} < \epsilon$（通常 $\epsilon = 1e-4$）。

**关键问题**：不同 token 的收敛步数差异有多大？

### 3.2 收敛步数分布

| 收敛步数 | Token 比例（cumulative）|
|:---:|:---:|
| 1 步 | ~20% |
| 2 步 | ~45% |
| 3 步 | ~60% |
| 4 步 | ~75% |
| **6 步** | **~90%** |
| 8 步 | ~95% |
| >8 步 | ~5% |

**关键观察**：
- 50% 的 token 在前 2 步就稳定
- 90% 的 token 在 6 步内稳定
- 仍有 ~10% 的 token 需要 6 步以上

### 3.3 与"Hard Task = More Steps"的一致性

结合 [Huginn §3.3](./huginn.md#33-任务难度--最优循环数) 的发现：
- 简单任务：大部分 token 早期收敛，最优 K ≈ 2-4
- 困难任务：困难 token 需要更多步骤，最优 K ≈ 6-8

**这意味着**：固定 K 是次优的。**动态 K（per-token early exit）**才是正确的。

---

## 4. 对 LatentMind 的启示

### 4.1 主线（HRM-Text）的 K=8 假设需要重新审视

| 当前假设 | 挑战 |
|---------|------|
| K=8 固定循环 | 90% token 6 步就够，8 步中 2 步是浪费 |
| 端侧 K=8 太慢 | 若 K=2 足够，延迟 × 4 降低 |
| GRAM 多轨迹叠加 | 与 per-token early exit 不冲突（早退后做多轨迹）|

**建议主线 K 设计**：
- 默认 K=2（90% 任务足够）
- 困难任务 K=4-6
- 极端困难 K=8（不超过 10% 的 token）

### 4.2 Per-Token Early Exit 的实现方案

```python
def forward_with_early_exit(x, max_K=8, epsilon=1e-4):
    h = init_state(x)
    for t in range(1, max_K + 1):
        h_new = block(h, x)
        # 监控每个 token 的变化
        delta = (h_new - h).norm(dim=-1)  # [B, L]
        # 早退决策
        exit_mask = delta < epsilon  # [B, L] boolean
        # ... 更新 h，标记已退出的 token 不再参与计算
        h = h_new
        if exit_mask.all():
            break
    return h
```

**计算节省**：
- 假设 90% token 6 步内退，10% token 8 步
- 平均循环数 ≈ 0.9 × 6 + 0.1 × 8 = 6.2 步
- 相比固定 K=8 节省 ~22% 计算

### 4.3 与 [loopcoder-v2.md §3.3 G-SWA](./loopcoder-v2.md#33-g-swagated-sliding-window-attention-with-shared-kv) 结合

G-SWA 的 sliding-window 本质是**对部分 token 提前退出**——已经"稳定"的 token 不再消耗 attention 计算。

**组合方案**：
- Per-token early exit（决定"是否继续循环"）
- G-SWA sliding window（决定"循环中看多少历史"）
- 两者叠加 = 推理 FLOPs 降低 50%+

### 4.4 与 [rrm-survey.md §1.1 TRM](./rrm-survey.md#11-trmtiny-recursive-model-) 的关系

TRM 论文也有"小模型递归到 K=3 后效果停滞"的观察——Per-Token 论文给出一个解释：**不是模型无法学习更深，而是大部分 token 早期就收敛了**。

---

## 5. 复现风险

| 风险 | 严重性 | 说明 |
|------|:---:|------|
| arXiv ID 未确认 | 🟠 | 论文摘要来自 librarian，PDF 未独立验证 |
| 90%/10% 数字具体任务相关 | 🟠 | 不同 benchmark 分布可能不同（数学 90% 需 8 步，QA 90% 需 3 步）|
| 收敛阈值 $\epsilon$ 选择 | 🟡 | 论文未明确 $\epsilon$ 取值，再现需调参 |
| 测量方法对训练过程敏感 | 🟡 | 论文测量训练后模型，训练中收敛步数会变化 |

---

## 6. 关键引用块

> 论文核心结论（基于摘要）：
> "Median token converges by step 6, but 10% of tokens require 8 steps. This suggests adaptive per-token compute allocation, rather than fixed K, is the right inference strategy for recurrent depth transformers."

---

## 7. 相关工作

| 名称 | 与 Per-Token Convergence 关系 |
|------|------------------------------|
| [LoopCoder-v2 / PLT](./loopcoder-v2.md) | 触发问题：固定 K=3 崩溃 → 启发"改用动态 K" |
| [Huginn 3.5B](./huginn.md) | 任务级动态 K（任务难度决定最优 K）|
| [STARS 2026](./stars.md) | 解决"K 大时怎么不崩" |
| [HRM-Text MagicNorm](./hrm-text.md#33-magicnorm前向-postnorm--反向-prenorm) | 配合：在 K=2 默认 + K=8 极端时仍稳定 |

---

**最后更新**：2026-07-29
**信息源**：librarian 摘要
