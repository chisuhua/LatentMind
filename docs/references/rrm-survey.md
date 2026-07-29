# RRM-B 谱系调查（Recurrent Reasoning Models）

> **范围**：除 [HRM-Text](./hrm-text.md) 和 [GRAM](./gram.md) 之外，HRM-B 家族（循环/递归推理模型）其他成员及"下一代架构"激进探索
> **创建日期**：2026-06
> **最后更新**：2026-06-13
> **整理目的**：补充 [hrm-original.md](./hrm-original.md)、[trm.md](./trm.md)、[gram.md](./gram.md) 的 RRM 谱系全景

---

## ⚠️ 重要说明

本文档不是 HRM-Text / GRAM 的替代品，而是其**谱系全景**。如要了解 LatentMind 直接依赖的 2 个核心架构（HRM-Text backbone + GRAM v1.5 扰动），请读：
- **[hrm-text.md](./hrm-text.md)** — 1.15B 文本版 backbone
- **[gram.md](./gram.md)** — 变分随机化扰动

---

## 1. RRM-B 谱系其他成员

### 1.1 TRM（Tiny Recursive Model）⭐

- **论文**：[arXiv 2510.04871](https://arxiv.org/abs/2510.04871) *Less is More: Recursive Reasoning with Tiny Networks*
- **作者**：Alexia Jolicoeur-Martineau（Samsung SAIL Montréal）
- **架构特点**：与 HRM 复杂的"分层双时间尺度"不同，TRM 走"**极简路线**"——只保留 3 个核心元素：输入、当前答案和隐式推理状态
- **循环机制**：单网络 2 层在潜空间中对当前答案反复迭代改进
- **定位**：证明"极简递归"也能替代庞大参数堆叠；**7M 模型在 Sudoku-Extreme 上达到 87.4%（vs HRM 27M 的 55.0%）；在 ARC-AGI-1 上达到 44.6%（vs HRM 27M 的 40.3%）**
- **详细笔记**：[trm.md](./trm.md)

### 1.2 Looped Transformer（循环 Transformer）

- **论文**：[arXiv 2604.15259](https://arxiv.org/abs/2604.15259) *Stability and Generalization in Looped Transformers*
- **架构特点**：核心是"**参数复用**"——同一 Transformer 块循环执行多次
- **循环机制**：在深度/时间维度上内部循环，权重共享
- **定位**：不增加参数规模的前提下，通过增加计算时间提升模型能力；参数高效利用的早期探索
- **关联**：[Universal Transformers (arXiv 1807.03819)](https://arxiv.org/abs/1807.03819) 是更早期的"循环共享"思想原型

### 1.3 Hyperloop Transformer

- **论文**：[arXiv 2604.21254](https://arxiv.org/abs/2604.21254) *Hyperloop Transformers*（Zeitoun, Torroba-Hennigen, Kim）
- **架构特点**：在 Looped Transformer 基础上引入"**超连接（Hyper-connections）**"优化循环中间块（begin / middle / end 三段）
- **定位**：从 LM 参数效率角度让循环架构在自然语言处理中更稳定
- **与 §1.2 区别**：Looped TF（§1.2）只循环 + 共享权重；Hyperloop（§1.3）额外加跨层超连接以解决信息瓶颈和梯度衰减

### 1.4 SE-RRM（Symbol-Equivariant Recurrent Reasoning Models）

- **论文**：[arXiv 2603.02193](https://arxiv.org/abs/2603.02193)
- **代码**：[github.com/ml-jku/SE-RRM](https://github.com/ml-jku/SE-RRM)
- **机构**：ELLIS Unit Linz / LIT AI Lab / JKU
- **架构特点**：在架构层面通过**符号等变层**强制置换等变性 → 同一问题在符号/颜色置换下输出保证一致
- **数据效率**：**2M 参数**在 9×9 数独训练后可泛化到 4×4 和 16×16、25×25（现有 RRM 无法外推）
- **关键细节**：使用 `adam-atan2` 优化器（**与 HRM-Text 一致** — 可能存在共同作者关系）
- **定位**：显式编码对称性提升神经推理的鲁棒性与可扩展性

### 1.5 Huginn & Ouro

- **状态**：✅ **Huginn 3.5B 已找到（arXiv:2502.05171）— 详细笔记见 [huginn.md](./huginn.md)**
- **HRM-Text 论文 Table 4 列为基线**：
  - Huginn 3.5B: MMLU 31.4, MATH 12.6（远低于 HRM-Text 1B 的 60.7 / 56.2）
  - Ouro 1.4B: MMLU 67.4, MATH 22.4（MMLU 高于 HRM-Text 但 MATH 远低）
- **关键发现（详见 [huginn.md §4](./huginn.md#4-关键数字)）**：
  - Huginn 3.5B 在 GSM8K 上随循环数 K=1→K=50 **持续提升**（34.6 → 64.8），**无 K=3 崩溃**
  - **反例意义**：证明 LoopCoder-v2 的"Only Loop Once"是 PLT 架构特有现象，不是循环本身的固有属性
  - HRM-Text 1B 的 K=4-8 假设**仍可能成立**——层次化结构可能免疫崩溃
- **Ouro**：arXiv ID 仍待查

---

## 2. 下一代架构的更广泛探索

除 RRM-B 谱系外，还有一些旨在解决 LLM 根本痛点的"下一代架构"：

### 2.1 Google Titans（记忆架构）

- **核心痛点**：大模型"健忘"和上下文窗口限制
- **架构设计**：
  - Transformer + 注意力 = **短期记忆**（处理当前上下文）
  - 深度神经网络（非 RNN）= **长期记忆**（存储历史信息）
- **创新机制**：
  - "惊喜度指标"（专门关注意外词汇以决定哪些信息值得记住）
  - "自适应遗忘"（通过衰减机制让模型忘记不再需要的信息）

### 2.2 Liquid AI（液态网络）

- **核心原理**：基于**常微分方程（ODE）的连续流**，而非离散 Transformer 层
- **优势**：复杂度呈**线性**，具有"无限上下文"的理论愿景；对 CPU/NPU 等非传统 GPU 硬件非常友好
- **公司**：Liquid AI（独立公司，2023 成立，源自 MIT CSAIL）

### 2.3 SpikingBrain（类脑脉冲）

- **核心原理**：模拟生物大脑的**脉冲放电动态**
- **优势**：复杂度为线性/常数；百万级 Token 长序列上可加速 100 倍；极低功耗；可集成工具调用
- **典型场景**：超长序列 + 节能部署

### 2.4 商汤 NEO-Unify（原生多模态）

- **核心突破**：彻底砍掉视觉编码器（VE）和 VAE，不再通过"组件拼凑"实现感知与生成
- **设计**：直接以近乎无损的像素和文字作为**原生输入**，采用 MoT（混合变换器）架构
- **意义**：多模态 AI 从"模态连接"进化为"原生统一智能体"
- **项目**：商汤 SenseNova U1

---

## 3. 核心架构横向对比

| 模型/架构 | 所属流派 | 核心机制 | 解决痛点 | 适用场景 |
|---|---|---|---|---|
| **HRM-Text** | RRM-B | 双时间尺度 H/L 模块 | 训练算力浪费；深层递归不稳定 | 通用语言预训练 |
| **GRAM** | RRM-B | 概率多轨迹采样 | 确定性递归陷入局部最优（坏盆地）| 复杂组合推理（N 皇后、数独）|
| **TRM** | RRM-B | 极简递归精炼 | 模型参数冗余，小模型泛化差 | 结构化谜题推理 |
| **Looped / Hyperloop** | 循环架构 | 参数复用 | 参数效率低 | 提升 LM 的参数效率 |
| **SE-RRM** | RRM-B | 符号等变性 | 现有 RRM 不可外推到未见尺寸 | 跨尺寸数独 / 棋盘 |
| **Google Titans** | 记忆架构 | 短期 + 长期记忆 | 灾难性遗忘 + 长上下文限制 | 终身学习、超长上下文 Agent |
| **Liquid AI** | 连续流架构 | ODE 连续流 | Transformer 二次方复杂度 | 无限上下文愿景 + 边缘部署 |
| **SpikingBrain** | 类脑架构 | 脉冲放电 | GPU 能耗过高 + 长序列慢 | 百万级 Token 加速 + 极低功耗 |

---

## 4. 演进趋势总结

当前 AI 架构正在从"**单纯堆叠参数和 Token（Scaling Law）**"转向"**计算范式的重构**"：

- **若关注"如何让小模型具备深度思考和推理能力"** → HRM-Text、GRAM、TRM、SE-RRM 等 RRM-B 谱系
- **若关注"突破上下文限制和硬件能效"** → Titans、Liquid AI、SpikingBrain
- **若关注"原生多模态融合"** → 商汤 NEO-Unify（与 LatentMind v1.0 感知层思路一致）

---

## 5. 与 LatentMind 决策相关

| 决策点 | 本谱系提供的信息 |
|---|---|
| **v1.0 backbone** | HRM-Text 1.15B（已选定，见 [hrm-text.md](./hrm-text.md)）|
| **v1.5 多轨迹推理** | GRAM 扰动（已选定，见 [gram.md](./gram.md)）|
| **是否考虑 TRM 风格简化** | 🟡 100M 规模消融后可考虑（TRM 优势主要在 <100M 符号任务）|
| **感知层是否走 NEO-Unify 路线** | 🟢 可参考"无独立视觉编码器"思想，v1.0 感知层 3 层 CNN 已体现 |
| **量化 / 部署相关** | Liquid AI / SpikingBrain 是非 GPU 部署的远期方向 |

---

## 6. 已知未解决问题

- [x] **Huginn 3.5B arXiv ID** → 已找到 2502.05171，详细笔记见 [huginn.md](./huginn.md)（2026-07-29 完成）
- [ ] Ouro 1.4B arXiv ID（HRM-Text Table 4 引用，标题搜索仍未命中）
- [ ] PTRM（删除）：之前的草稿提到"PTRM"作为 RRM-B 成员之一，但 arXiv 1909.04610 "PTRM" 实为 *Perceived Terrain Realism Metrics*（地形真实感评估），与循环推理**无关**。**已删除避免误导**。
- [ ] Looped Transformer 的早期论文（非 Hyperloop 版本）
- [ ] Liquid AI / SpikingBrain 的具体公开模型/代码链接

---

## 7. 2026-07-29 增补：LoopCoder-v2 冲击波

> 完整论证见 [loopcoder-v2.md](./loopcoder-v2.md)。本节仅补充对 RRM-B 谱系的影响。

**新增谱系成员**：

| 名称 | 谱系位置 | 关键贡献 |
|------|---------|---------|
| [LoopCoder-v2 / PLT](./loopcoder-v2.md) | RRM-B + 并行循环 | "Only Loop Once" + 3 个崩溃机制诊断 |
| [STARS 2026](./stars.md) | RRM-B 修复方案 | LayerNorm 根因 + Jacobian 谱半径正则化（修复 12.21pp）|
| [Per-Token Convergence](./per-token-convergence.md) | RRM-B 动态深度 | 90% token 6 步收敛，10% 需要 8 步 |

**对 RRM-B 谱系整体的影响**：
1. "循环次数越多越好"是**迷信**——但**崩溃是架构特异性的**，不是循环本身的属性
2. Huginn（无 CLP 顺序循环）vs PLT（有 CLP 扁平循环）是**对照实验**——CLP + 共享参数是 PLT 崩溃的根因之一
3. HRM-Text（层次化 + 不同参数）的 K=4-8 假设**需要独立验证**——见 [RDD-0001 RFC](../rfcs/RDD-0001-k-sweep-experiment.md)

---

**最后更新**：2026-06-13
**信息源**：
- [arXiv 2505.14674](https://arxiv.org/abs/2505.14674) (RRM-A 作者核查)
- [arXiv 2510.04871](https://arxiv.org/abs/2510.04871) (TRM)
- [arXiv 2506.21734](https://arxiv.org/abs/2506.21734) (HRM 原始 27M)
- [arXiv 2605.20613](https://arxiv.org/abs/2605.20613) (HRM-Text)
- [arXiv 2605.19376](https://arxiv.org/abs/2605.19376) (GRAM)
- [arXiv 2603.02193](https://arxiv.org/abs/2603.02193) (SE-RRM) + [github.com/ml-jku/SE-RRM](https://github.com/ml-jku/SE-RRM)
- [arXiv 2604.21254](https://arxiv.org/abs/2604.21254) (Hyperloop Transformer)
- 之前 librarian 报告 + HRM-Text 论文 Table 4（Huginn / Ouro 引用）
