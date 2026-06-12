# GRAM（Generative Recursive reAsoning Models）参考资料

> **论文**：*Generative Recursive reAsoning Models (GRAM)*
> **作者**：Junyeob Baek, Mingyu Jo, Minsu Kim, Yoshua Bengio, Sungjin Ahn
> **机构**：KAIST, NYU, Mila
> **arXiv**：2605.19376（2026-05-22）
> **会议**：ICLR 2026 Workshop RSI Poster
> **项目页**：https://ahn-ml.github.io/gram-website
> **OpenReview**：https://openreview.net/forum?id=Vxu6kcIjwV
> **本地源**：[`papers/arxiv-2605.19376-gram.pdf`](./papers/arxiv-2605.19376-gram.pdf) (1.3 MB)
> **本地 HTML**：[`papers/arxiv-2605.19376-gram.html`](./papers/arxiv-2605.19376-gram.html) (380 KB)

---

## 1. 一句话定位

**"概率多轨迹递归推理"**：把 HRM/TRM 风格的递归推理**从确定性单轨迹改为随机多轨迹**——每次循环步对潜在状态加学习的高斯扰动，用变分 ELBO 训练，推理时**多轨迹并行采样**生成多个候选解。**LatentMind v1.5 第二个核心依赖**。

---

## 2. 摘要原文翻译

> 未来神经推理系统应如何实现扩展计算？递归推理模型（RRM）通过**共享转移函数**对**持久潜在状态**做迭代精炼，提供了对自回归序列扩展的有前途的替代方案。但现有 RRM 主要是**确定性的**：遵循单条潜在轨迹、收敛到单一预测。我们提出 **GRAM**——一个**将递归潜在推理变为概率多轨迹计算**的框架。GRAM 将推理建模为**随机潜在轨迹**，支持多假设、替代解题策略，以及**通过递归深度和并行轨迹采样**两条轴的测试时 scaling。这产生一个**潜变量生成模型**：支持条件推理 p_θ(y|x)，以及无条件生成 p_θ(x)。GRAM 用**摊销变分推断**训练，在结构化推理与多解约束满足任务上超越确定性循环/递归基线，并展示无条件生成能力。

---

## 3. 核心创新

### 3.1 从确定性到概率

| 维度 | HRM / TRM（确定性）| GRAM（概率）|
|---|---|---|
| 轨迹 | 单条确定性轨迹 | **多条随机轨迹** |
| 多解支持 | ❌ | ✅（N-Queens 等多解任务） |
| 推理时 scaling | 仅深度（更多循环步）| **深度 + 宽度**（并行采样多轨迹）|
| 训练 | 监督 / CE 损失 | **变分 ELBO** |
| 不确定性建模 | 无 | 通过学习的高斯扰动 |

### 3.2 关键设计：学习的高斯扰动

- 每次递归步对潜在状态加扰动：ε ~ N(μ_θ(u), σ²_θ(u) I)
- **μ_θ, σ_θ 是学习的小网络**（输入当前状态 u，输出扰动的均值和方差）
- 不同输入 → 不同扰动分布 → 多样化轨迹

### 3.3 变分 ELBO 训练

GRAM 是**潜变量概率模型 p_θ**，完整轨迹 τ = (z_0 → z_1 → ... → z_T_Total) 是一串潜变量：
- 优化目标：max ELBO wrt 生成参数 θ 和变分参数 φ
- 重参数化技巧 → 可微采样
- 训练时执行 T × N_sup 步（深度监督）

### 3.4 测试时 scaling：两条轴

- **深度轴**：增加 T（更多循环步）→ 更深推理
- **宽度轴**：并行采样 K 条轨迹 → K 个候选解
- 可同时使用：T × K 的二维 scaling 空间

### 3.5 同时支持有条件与无条件生成

- 条件 p_θ(y|x)：给定输入 x，生成输出 y
- 无条件 p_θ(x)：不输入或固定输入，生成 x 本身
- 同一框架，无需两种模型

---

## 4. 与 HRM / TRM 的关系

```
HRM（2025-06，27M）
  │
  │ 继承分层循环
  ▼
TRM（2025-10，5M）
  │
  │ 简化结构（单网 2 层 + 多步 BPTT）
  ▼
GRAM（2026-05，10-11M）
  │
  │ 加上随机扰动 + 多轨迹 + ELBO
  ▼
LatentMind v1.5 计划：
  HRM-Text 1.15B backbone
    + GRAM 风格的随机扰动（v_H residual）
    + GRAM 风格的多轨迹并行推理
```

**关键论文引用**：
- GRAM 引用 HRM [8] 和 TRM [9] 为"structured reasoning" 早期证据
- GRAM 强调"deterministic recurrent models"是当前 RRM 的局限
- GRAM 也引用 Universal Transformer [10] 和 Looped Transformer [7] 为"shared Transformer block 重复应用"的设计先驱

---

## 5. 实验任务与结果

### 5.1 任务列表

| 任务 | 类型 | 评估能力 |
|---|---|---|
| **Sudoku-Extreme** | 约束满足 | 结构化推理、约束传播 |
| **ARC-AGI** | 抽象变换 | 通用智能核心基准 |
| **N-Queens** | 多解约束 | 多解恢复、轨迹多样性 |
| **Graph Coloring** | 多解约束 | 多解恢复 |
| **binarized MNIST** | 无条件生成 | 生成能力、p_θ(x) 路径 |

### 5.2 关键结果（论文 Method 段表格）

**Direct Pred 基线 vs GRAM**（Sudoku-Extreme / ARC-AGI 任务）：

| 模型 | Rec. | Gen. | # Params | 准确率 | 覆盖率 | 冲突↓ |
|---|---|---|---|---|---|---|
| Direct Pred (8 layers) | ✗ | ✗ | 27M | 40.4 ± 1.1 | 13.7 ± 1.1 | 13.6 ± 0.5 |
| Direct Pred (32 layers) | ✗ | ✗ | 100M | 40.2 ± 1.3 | 13.6 ± 1.1 | 13.1 ± 0.4 |
| ... | | | | | | |
| **GRAM** | ✅ | ✅ | ~10-11M | **显著超越** | **更高** | **更低** |

> 详细准确率数字待深度研究返回后补充
> Rec. = 循环推理；Gen. = 多轨迹生成

### 5.3 关键可视化（论文 Figure 18, 19）

- **Figure 18（TRM）**：单条确定性路径，无跳出次优轨迹能力
- **Figure 19（GRAM, 50 样本）**：50 条不同采样轨迹，**分散到达多个解**
- 直观证明 GRAM 的多轨迹优势

### 5.4 训练成本

- **硬件**：8 × NVIDIA RTX Pro 6000
- **训练**：200K epochs，batch size 768
- **对比**：Looped TF 在 Sudoku-Extreme 上 8× Pro 6000 跑 19h（说明 GRAM 训练不便宜但可行）

---

## 6. ⚠️ 关键限制（与 LatentMind 相关）

### 6.1 论文自承认的限制

> "the sequential nature of deep supervision limits training efficiency compared to Transformers, **posing a significant barrier to scaling GRAM toward larger foundation models**"

**翻译**：深度监督的串行特性限制了训练效率，**对将 GRAM 扩展到更大基础模型构成重大障碍**。

### 6.2 实际验证的规模上限

- 论文实验**仅跑到 10–11M 参数**（Sudoku-Extreme、ARC-AGI、N-Queens）
- **1B 规模完全未验证**

### 6.3 对 LatentMind v1.5 的影响

| 风险 | 严重性 | 缓解策略 |
|---|---|---|
| GRAM 风格扰动 + 1.15B HRM-Text 可能不收敛 | 🔴 高 | **小规模消融先行**：100M 参数先验证 |
| 1B 变分训练不稳定（KL 散度爆炸、潜空间坍缩）| 🔴 高 | KL 退火、free bits、β-VAE 风格调度 |
| 多轨迹推理计算量 × 轨迹数 | 🟡 中 | 轨迹数 K 可配置（2-4 条作默认）|
| 无条件生成路径 p_θ(x) 在多模态场景意义不明 | 🟡 中 | v1.5 可暂不用，只用条件路径 |
| 推理时 +1× 随机化层延迟 | 🟢 低 | 重叠采样（overlap sampling）|

---

## 7. 与 LatentMind v1.5 架构的集成点

### 7.1 推荐集成位置

GRAM 的扰动注入在**每次递归步**，对 LatentMind v1.5（HRM + GRAM 双层叠加）来说：

```
H 模块 residual:
    z_H_new = H_module(z_H + z_L + ε_H)   ← 在 z_L 加入之后、H 模块输出之前加扰动
    ε_H ~ N(μ_θ(z_H + z_L), σ²_θ(z_H + z_L) I)

L 模块 residual（可选）:
    z_L_new = L_module(z_L + z_H + ε_L)
    ε_L ~ N(μ'_θ(z_L + z_H), σ²'_θ(z_L + z_H) I)
```

### 7.2 训练目标

```
ELBO = E_q_φ[log p_θ(y | x, τ)]  -  β · KL(q_φ(τ | x, y) || p_θ(τ))
      = 响应 NLL（与 HRM-Text 任务完成目标相同）
      - KL 散度（让扰动分布不过度发散）
```

### 7.3 推理模式

- **单轨迹模式**（默认）：ε ~ N(0, 0)（或 μ_θ + σ_θ × 0） → 退化为近似 HRM-Text
- **多轨迹模式**：采样 K 次，取最佳或平均
  - 评估"安全"度：多轨迹方差小 → 高置信；方差大 → 不确定

### 7.4 消融实验设计

| 阶段 | 规模 | 对照 | 目标 |
|---|---|---|---|
| 消融 1 | 100M | HRM-Text 风格 vs HRM-Text + GRAM 扰动 | 验证扰动在语言任务上有益 |
| 消融 2 | 100M | 单轨迹 vs 多轨迹（K=2,4,8）| 验证多轨迹在不确定场景下有效 |
| 消融 3 | 1B | 通过消融 1/2 后再扩 | 验证 1B 可扩展性 |

---

## 8. 待补充

> ⚠️ 本节是**初步笔记**，基于论文 HTML 摘录 + 之前 librarian 报告。详细数字（完整 Table、ELBO 系数、超参数）需要**深度研究**（后台任务 `bg_eceb4e2e`）返回后补充：

- [ ] 完整消融表（论文 Table 1-3）
- [ ] ELBO 的 β 系数与退火 schedule
- [ ] μ_θ, σ_θ 网络结构（MLP/Transformer 块？）
- [ ] T × N_sup 的具体数值
- [ ] N-Queens 多解恢复的具体数字
- [ ] 1B 规模是否有任何讨论/外推
- [ ] OpenReview 评审意见
- [ ] 是否有任何 GRAM-style 应用于 LLM 的工作

---

## 9. 关键引用块

> 摘要核心创新：
> "GRAM models reasoning as a stochastic latent trajectory, enabling multiple hypotheses, alternative solution strategies, and inference-time scaling through both recursive depth and parallel trajectory sampling."

> ELBO 训练：
> "Trained with amortized variational inference, GRAM improves over deterministic recurrent and recursive baselines on structured reasoning and multi-solution constraint satisfaction tasks."

> 限制：
> "the sequential nature of deep supervision limits training efficiency compared to Transformers, posing a significant barrier to scaling GRAM toward larger foundation models."

> 与 HRM 的对比（N-Queens Figure）：
> "Prior RRMs (e.g. HRM, TRM) are deterministic—all runs collapse to an identical trajectory, converging to a single solution and failing to explore alternatives, while GRAM explores diverse trajectories, producing diverse trajectories that reach multiple solutions."

---

## 10. 与 LatentMind 决策树

| 决策 | 建议 |
|---|---|
| **v1.0**（仅 HRM-Text backbone）| 不引入 GRAM 扰动 → 走确定路径 |
| **v1.5**（HRM + GRAM）| ✅ 引入扰动 + 多轨迹 + ELBO |
| **扰动注入位置** | H 模块 residual（不推荐 L 模块，先稳） |
| **多轨迹数 K** | 训练时 K=1（单轨迹 ELBO 训练）→ 推理时 K=2~4（多假设） |
| **ELBO β 系数** | 1.0 + KL 退火 0→1 跨越 10K 步 |
| **最大规模验证** | 先 100M → 通过 → 1B |
| **无条件生成路径** | v1.5 暂不用（与多模态不直接对应）|

---

**最后更新**：2026-06-13（v1 初步版，深度研究回来后 v2 增强）
**信息源**：arXiv 2605.19376 全文（HTML + PDF）、之前 librarian 报告、HRM/TRM 对比分析
