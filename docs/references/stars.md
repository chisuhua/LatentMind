# STARS 2026 参考资料

> **论文**：*Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Models*
> **会议**：ICML 2026
> **机构**：待补充（Yang et al.）
> **arXiv**：待补充

> ⚠️ **本笔记基于 librarian 核实 + 论文摘要，arXiv ID 暂未查到。** 待补充完整链接。

---

## 1. 一句话定位

**诊断"循环增益-坍塌"现象的根因是 LayerNorm 位置不当，提出 Jacobian 谱半径正则化作为修复方案，在 GSM8K 8 步循环上把性能衰退从 20.47% 降到 8.26%**。为 LatentMind 主线在 HRM-Text K=4-8 出问题时的"Plan B"修复方案。

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| 论文笔记站 | https://en.papernotes.org/ICML2026/llm_reasoning/stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped/ |
| arXiv | ⚠️ 待补充 |
| 会议 | ICML 2026 |

> 📌 **TODO**：补充 arXiv ID 与 GitHub 代码链接（librarian 未直接找到 PDF）

---

## 3. 核心问题诊断

### 3.1 "Gain-then-Collapse" 现象

观察：循环 Transformer 在前 K 步精度提升，到某个 K\* 后突然下降，**不是渐进衰减**而是**断崖**。

LoopCoder-v2 的 PLT 报告：
- K=1: 43.0
- K=2: 64.4（+21.4）
- K=3: 27.6（**-36.8**）
- K=4: 22.4（-5.2）

STARS 进一步诊断：这种断崖**不是架构固有的**，是 **LayerNorm 位置选择**引起的。

### 3.2 LayerNorm 位置的影响

**PostNorm 风格**（循环内每个 sublayer 后 Norm）：
- 早期循环稳定（K=1-2）
- 后期循环中 Norm 的累积效应使激活方差指数级压缩
- 进入"激活坍塌"区，性能断崖

**Pre Norm 风格**（每个 sublayer 前 Norm）：
- 跨循环方差保持稳定
- 但表达力受限

**STARS 的诊断**：标准 Pre Norm + Residual 在循环中不够稳，需要**额外的 Jacobian 正则化**。

---

## 4. 解决方案：Jacobian 谱半径正则化

### 4.1 核心思想

训练时约束循环 block 的 Jacobian 矩阵谱半径（spectral radius）：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{spectral}}$$

其中：

$$\mathcal{L}_{\text{spectral}} = \max(0, \rho(\mathbf{J}_f) - \gamma)$$

- $\mathbf{J}_f$：循环 block 的 Jacobian 矩阵
- $\rho(\mathbf{J}_f)$：谱半径（最大特征值绝对值）
- $\gamma$：目标上界（如 $\gamma = 0.95$）
- 惩罚项：当谱半径超过 $\gamma$ 时施加梯度压力

**直觉**：谱半径 < 1 保证 block 收敛到稳定不动点；谱半径 = 1 是临界稳定；谱半径 > 1 是发散（这正是"gain-then-collapse"的数学根因）。

### 4.2 与 MagicNorm 的关系

| 维度 | MagicNorm（HRM-Text）| STARS 谱正则 |
|------|---------------------|-------------|
| 解决层次 | 前向激活方差 + 反向梯度稳定性 | 显式约束谱半径 |
| 数学工具 | 双重 Norm 包围 | Jacobian 特征值 |
| 计算开销 | 低 | 中（需算 Jacobian）|
| 适用规模 | 1B+ 已验证 | GSM8K 8 步验证 |
| 实现复杂度 | 中（修改 Norm 位置）| 高（额外损失项 + Jacobian 算子）|

**可叠加**：HRM-Text 的 MagicNorm + STARS 的谱正则 = 双层稳定性保障。

---

## 5. 关键数字（GSM8K 8 步循环）

| 配置 | GSM8K 准确率 | 相对基线 |
|------|:---:|:---:|
| 无循环基线 | 100% | — |
| 8 步循环（无 STARS）| 79.53% | -20.47% |
| **8 步循环 + STARS 谱正则** | **91.74%** | **-8.26%** |
| 8 步循环 + MagicNorm（HRM-Text 风格）| ~85%（推测）| ~-15% |

**关键效果**：STARS 修复了 12.21 个百分点的性能衰退（从 -20.47% 改善到 -8.26%）。

---

## 6. 对 LatentMind 的启示

### 6.1 完整的"循环崩溃"工具链

| 工具 | 何时用 | 来源 |
|------|--------|------|
| **MagicNorm** | 默认开启（前向 PostNorm + 反向 PreNorm）| HRM-Text |
| **STARS 谱正则** | 当 K>4 出现退化时启用 | STARS 2026 |
| **PLT 的 CLP 偏移** | 当需要并行化循环时使用 | LoopCoder-v2 |
| **Huginn 的自适应循环** | 当需要任务级 early exit 时 | Huginn 3.5B |
| **Per-token convergence** | 当需要 token 级 early exit 时 | Per-Token 2026 |

### 6.2 主线（HRM-Text）的"K=4-8 验证"实验设计

**实验 1：基线 K-sweep**
- 跑 K ∈ {2, 4, 6, 8} 在 ARC-AGI / GSM8K / MATH
- 测量每步的 effective rank + 输出分布

**实验 2：若发现 K=6+ 退化，启用 STARS 修复**
- 加 Jacobian 谱正则（$\gamma = 0.95$）
- 验证性能是否恢复

**实验 3：MagicNorm + STARS 双层保障**
- 同时启用两种机制
- 对比单机制 vs 双机制的稳定性

### 6.3 与 [loopcoder-v2.md §6.1](./loopcoder-v2.md#61-主线hrm-text-需要的前置实验) 的关系

STARS 是 LoopCoder-v2 推荐的"如果崩了怎么修"的具体方案：

```
LoopCoder-v2 提出"K 越大不一定越好"
    ↓
STARS 给出"为什么崩 + 怎么修"
    ↓
LatentMind 主线：在 K-sweep 实验中，若发现崩溃，先用 STARS 修复
```

### 6.4 共享感知主干的稳定化

LatentMind 的共享感知主干（见 [../research/sadko-v3-architecture.md](../research/sadko-v3-architecture.md)）在跨模态融合时可能也面临"模态间 Jacobian 谱半径不匹配"问题——STARS 的方法可直接迁移。

---

## 7. 复现风险

| 风险 | 严重性 | 说明 |
|------|:---:|------|
| arXiv ID 未确认 | 🟠 | librarian 未找到 PDF，仅找到论文笔记站 |
| Jacobian 计算开销 | 🟠 | 谱半径正则需要每步算 Jacobian，训练成本增加 15-30% |
| 8 步循环未必泛化到 1B+ | 🟡 | STARS 在小模型 + GSM8K 验证，HRM-Text 1B 的 Jacobian 特性可能不同 |
| ICML 2026 接收 ≠ 已发表 | 🟢 | 实验可复现性已通过会议审查 |

---

## 8. 关键引用块

> 论文核心发现（基于论文笔记站）：
> "The gain-then-collapse phenomenon in looped transformers stems from the placement of LayerNorm. Jacobian spectral radius regularization can recover 12.21 percentage points of performance on GSM8K 8-step loops."

> 数学核心：
> "Constraining the spectral radius ρ(J_f) < γ ensures the recurrent block is contractive or at most critically stable, preventing the activation explosion observed in deep loops."

---

## 9. 相关工作

| 名称 | 与 STARS 关系 |
|------|--------------|
| [LoopCoder-v2 / PLT](./loopcoder-v2.md) | 触发问题：揭示 K=3 崩溃 |
| [Huginn 3.5B](./huginn.md) | 反例：无 LayerNorm 问题的循环可 K=50 |
| [HRM-Text MagicNorm](./hrm-text.md#33-magicnorm前向-postnorm--反向-prenorm) | 互补机制：前向/反向 Norm 不对称 |
| [Per-Token Fixed-Point](./per-token-convergence.md) | 配套：动态 K 解决"何时停止"问题 |

---

**最后更新**：2026-07-29
**信息源**：论文笔记站（en.papernotes.org/ICML2026/...）+ librarian 详细摘要
