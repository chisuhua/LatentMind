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
- 协方差是**对角**（per-dimension 独立），不是全协方差矩阵

### 3.3 μ_θ / σ_θ 网络结构（论文 §B.1 Table 6）

**是 SwiGLU MLP，不是 Transformer block**：

| 网络 | 角色 | 结构 |
|---|---|---|
| μ_θ(u_t) | prior 均值 | SwiGLU MLP（per-token） |
| σ_θ(u_t) | prior 标准差 | SwiGLU MLP（per-token） |
| μ_φ(u_t, y) | posterior 均值 | SwiGLU MLP（per-token，额外输入 y 编码） |
| σ_φ(u_t, y) | posterior 标准差 | SwiGLU MLP（per-token） |

- **4 个独立 SwiGLU MLP**（prior 2 + posterior 2）
- **per-token、position-wise**（不是 sequence-level）
- 社区默认 `hidden_mult = 1`（即 d_hidden = D = 512，与 backbone FFN 同宽）
- 输入 u_t 维度 = h_t 维度 = D

### 3.4 变分 ELBO 训练

GRAM 是**潜变量概率模型 p_θ**，完整轨迹 τ = (z_0 → z_1 → ... → z_T_Total) 是一串潜变量：
- 优化目标：max ELBO wrt 生成参数 θ 和变分参数 φ
- 重参数化技巧 → 可微采样
- 训练时执行 T × N_sup 步（深度监督）

**ELBO 形式**（论文 §B.2）：

$$\mathcal{L}_{ELBO} = \mathbb{E}_{q_\phi(\tau|x,y)}[\log p_\theta(y|x,\tau)] - \beta \cdot \mathcal{L}_{KL}$$

- 第一项：响应端 NLL（与 HRM-Text 任务完成目标一致）
- 第二项：KL 散度正则化
- **β 是 per-task 调的超参数**（不是固定值！）

**KL 平衡（Dreamer-V2 风格）**：

$$\mathcal{L}_{KL} = \alpha \cdot KL(\text{sg}(q_\phi)\|p_\theta) + (1-\alpha) \cdot KL(q_\phi\|\text{sg}(p_\theta))$$

- **α = 0.8**（论文固定值）
- sg = stop gradient；前半段训练 prior 适配 posterior，后半段训练 posterior 适配 prior

### 3.5 per-task β 系数表（论文 §B.2 / Table 7 抓取）

| 任务 | β (KL 权重) |
|---|---|
| Sudoku-Extreme | 0.1 |
| ARC-AGI-1 | 0.04 |
| ARC-AGI-2 | 0.1 |
| N-Queens 8×8 | 0.07 |
| N-Queens 10×10 | 0.045 |
| Graph Coloring 8v | 0.5 |
| Graph Coloring 10v | 0.45 |
| Binarized MNIST | 0.07 |

**关键观察**：
- β 范围从 0.04（ARC-AGI-1）到 0.5（Graph Coloring 8v），**跨一个数量级**
- 图形结构任务需要更大 KL 压力
- ⚠️ **论文未明确说明 β 是否有 warmup / annealing schedule**（社区复现默认无 annealing）

### 3.6 训练深度（论文 §B.2）

| 符号 | 含义 | 值 |
|---|---|---|
| **N_sup** | 每个样本执行的深度监督步数 | **16**（所有任务统一）|
| **T** | 每个 supervision step 内的 transition 数 | ⚠️ **论文未明确** |
| **K** | 每个 transition 内的 low-level 更新数 | ⚠️ **论文未明确** |
| T_Total | 总步数 = T × N_sup | ⚠️ 由 T 决定 |

**对 LatentMind 集成的含义**：必须**自己拍板 T 和 K** 的默认值；社区复现 ad3002/gram 也列为"GAP"。

### 3.7 测试时 scaling：两条轴

- **深度轴**：增加 T（更多循环步）→ 更深推理
- **宽度轴**：并行采样 K 条轨迹 → K 个候选解
- 可同时使用：T × K 的二维 scaling 空间

### 3.8 同时支持有条件与无条件生成

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

## 8. 深度研究后的关键发现（v2 增强）

> 本节是 2026-06-13 完成的**深度研究**（基于论文全文 + 表格抓取 + 社区复现 + 二手分析）的关键发现。

### 8.1 论文 specification gap 清单

| 项 | 论文是否明确 | 默认值 / 推断 |
|---|---|---|
| **N_sup**（每样本深监督步数）| ✅ 16 | 16（所有任务） |
| **T**（每步 transition 数）| ❌ 未明确 | 社区 ad3002/gram 列为 GAP，需自行拍板 |
| **K**（每 transition low-level 更新）| ❌ 未明确 | 同上 |
| **β（KL 权重）per-task** | ✅ 表 7 给出 | 见 §3.5 表格 |
| **α（KL 平衡系数）** | ✅ 0.8 | 固定 |
| **KL annealing schedule** | ❌ 未明确 | 社区默认无 annealing |
| **μ_θ/σ_θ 网络结构** | ✅ SwiGLU MLP | hidden_mult=1, D=512 |
| **协方差形式** | ✅ 对角 | per-dimension 独立 |
| **μ_θ/σ_θ hidden_mult** | ❌ 未明确 | 社区默认 1 |

### 8.2 1B 扩展的真实状态

**论文 §5 限制原话**：
> "The sequential nature of deep supervision limits training efficiency compared to Transformers, **posing a significant barrier to scaling GRAM toward larger foundation models**."

**论文实际状态**：
- ✅ **0 个 1B 实验**（所有实验 ≤ 27M）
- ✅ **0 个 scaling law 拟合**
- ✅ **0 formal theory** 分析 stochastic trajectory 在大模型上是否 work
- ✅ **"未来工作"具体计划**：§5 中**没有明确的"we plan to do X"段落**（仅陈述限制）

**第三方分析（aiHola 报道）**：
> "A polite way of saying we have no idea if any of this works at 1B parameters, let alone 70B"

**对 LatentMind v1.5 的含义**：GRAM 1B 是**完全 open question**——v1.5 集成风险最高的环节，必须**先在 100M 规模消融**再扩到 1B。

### 8.3 OpenReview 状态

| 项 | 状态 |
|---|---|
| 会议 | ICLR 2026 Workshop RSI（Recursive Self-Improvement） |
| 接收形式 | Poster |
| 公开评审 | ❌ 0 个（Workshop poster 通常不公开 review） |
| 提交信息 | "Submission Number: 113", "Last Modified: 12 Mar 2026" |
| Rebuttal | ❌ 无（因为没有 review） |
| 非正式讨论 | aihola、arXivIQ、gentic.news、Moonlight、alphaXiv audio |

**对 v1.5 集成的含义**：**无正式评审可参考**，只能依赖：
- 作者 Twitter/X 自我宣传
- 第三方博客评论
- 1 个独立社区复现（ad3002/gram）

### 8.4 社区复现现状

- **ad3002/gram**（GitHub）：独立复现，**与论文结果接近**，但对 T/K/annealing 需自行拍板默认值
- 复现笔记中明确标出 "GAP"：哪些超参论文没说清楚

---

## 9. 待补充（v3 进一步工作）

- [ ] 完整消融表数字（论文 Table 1-3 各模型具体准确率）
- [ ] N-Queens 多解恢复的具体数字
- [ ] 不同任务下 ELBO 各分项的相对贡献
- [ ] 是否有任何 GRAM-style 应用于 LLM 的工作（截至 v2 没有公开工作）
- [ ] ad3002/gram 复现中拍板的 T/K 默认值是什么

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

**版本**：v2（深度研究增强版，2026-06-13）
**信息源**：
- arXiv 2605.19376 全文（HTML + PDF）
- 论文 §B.1 Table 6（网络结构）
- 论文 §B.2 / Table 7（per-task β 系数）
- 论文 §5（限制与未来工作原文）
- OpenReview Vxu6kcIjwV（公开信息）
- 社区复现 ad3002/gram（GitHub）
- 二手分析：aiHola、arXivIQ、Moonlight、alphaXiv
- 之前 librarian 报告 + 本次深度研究（session ses_1435d428affexXxvw97vlCFYFC）
