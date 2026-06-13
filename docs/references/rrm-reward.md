# RRM-A: Reward Reasoning Model

> **论文**：*Reward Reasoning Model*
> **作者**：Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei（7 人）
> **机构**：**清华大学**（注意：不是"微软/清华/北大联合"）
> **arXiv**：[2505.14674](https://arxiv.org/abs/2505.14674)
> **本地源**：（未下载）
> **创建日期**：2026-06
> **最后更新**：2026-06-13

---

## ⚠️ 重要命名歧义

> **"RRM" 在当前文献中存在 2 种完全不同的含义**：
>
> | 缩写 | 全称 | 代表项目 | 本质 |
> |---|---|---|---|
> | **RRM-A** | **Reward Reasoning Model** | 清华 RRM（本文档）| **功能架构**：让模型学会像裁判一样思考的奖励模型 |
> | **RRM-B** | **Recurrent Reasoning Model** | HRM, TRM, SE-RRM | **架构家族**：循环神经网络的统称 |
>
> HRM-Text 属于 RRM-B。本文档讨论的是 RRM-A。

---

## 1. 一句话定位

**用"思维链推理 + 动态计算分配"重构奖励建模**：把奖励模型从"输入 Query+Response → 标量分数"重构为"先自回归生成 CoT 推理链 → 输出结构化偏好判断"，并支持 RL（GRPO）+ Rule-Based Reward 训练。

---

## 2. 与传统奖励模型对比

| 维度 | 传统奖励模型 (Scalar RM) | RRM-A (Reward Reasoning Model) |
|---|---|---|
| **输入** | Query + Response | Query + Response A + Response B |
| **处理过程** | 直接前向传播 → 标量分数 | **自回归生成 CoT 推理链** → 最终判断 |
| **输出** | 单一浮点数（如 0.85） | 结构化文本（思考过程 + `\boxed{Assistant A}`） |
| **计算模式** | 固定 FLOPs | **自适应 Test-Time Compute**（难任务想更久） |
| **训练方式** | SFT / Bradley-Terry Loss | **RL (GRPO) + Rule-Based Reward** |
| **基座依赖** | 任意 Encoder/Decoder | Decoder-only LLM（Qwen2, DeepSeek-R1） |

---

## 3. 核心架构组件

### 3.1 显式推理引擎

- 模型在给出偏好判断前，必须先生成一段包含"比较"、"反思"、"验证"等关键词的思维链
- 这段 CoT **不是人工标注**的，而是通过 RL 自主演化出来的

### 3.2 规则化奖励环境

- 训练时不依赖昂贵的人工推理标注
- 构建**基于规则的奖励函数**（选对 = +1，选错 = -1）
- 在 GRPO 算法下让模型试错学习推理策略

### 3.3 多响应评估策略

| 策略 | 复杂度 | 用途 |
|---|---|---|
| ELO 循环赛 | $O(n^2)$ | 完整排序 |
| 淘汰赛制 | $O(n)$ | Best-of-N 采样 |
| 多数投票 | — | 结合并行扩展提升鲁棒性 |

### 3.4 Test-Time Compute Scaling 机制

支持两种扩展模式：
- **序列扩展**：增加 thinking budget（推理链长度）
- **并行扩展**：增加成对比较次数或投票采样数

---

## 4. 开源项目与资源

| 项目 | 基座 | 参数量 | 链接 | 备注 |
|---|---|---|---|---|
| **官方 RRM** | Qwen2 | 7B / 14B / 32B* | [arXiv 2505.14674](https://arxiv.org/abs/2505.14674) | 清华原版（**作者列表需进一步核查具体 size 映射**）|
| **RM-R1** | Qwen2.5 | 7B / 14B | 社区复现 | 强化推理迁移能力 |
| **Think-J** | Qwen2 | 7B | HuggingFace | 专注 Judge 任务的推理增强版 |
| **SE-RRM** | Custom | **2M** | [GitHub: ml-jku/SE-RRM](https://github.com/ml-jku/SE-RRM) | ⚠️ **同名不同义**：指 Recurrent Reasoning Model（RRM-B），见 [rrm-survey.md](./rrm-survey.md) |

---

## 5. 性能声明（需进一步验证）

> ⚠️ RRM-A 自称在 **RewardBench 上达到 SOTA**。本笔记尚未独立验证具体数字与对比基线。
>
> **建议下一步**：
> - 查 [RewardBench 排行榜](https://huggingface.co/spaces/allenai/reward-bench) 对比 RRM 与 Skywork、InternRM、Nemotron 等同期模型
> - 在 [projects.md §2.5](./projects.md) 中查 minimind 项目的偏好对齐基线
> - 在 [trm.md §3.3](./trm.md) 中查 RRM 谱系对比

---

## 6. 与 LatentMind 项目的关系

### 6.1 vs. HRM-Text（[hrm-text.md](./hrm-text.md)）

| 维度 | HRM-Text | RRM-A |
|---|---|---|
| 创新层级 | **基础网络拓扑**（新架构）| **训练范式 + 推理流程**（新功能）|
| 核心目标 | 突破 Transformer 推理深度瓶颈 | 提升奖励模型的判断准确性与可解释性 |
| 底层结构 | 双时间尺度分层循环模块（非 Transformer）| 标准 Decoder-only Transformer（Qwen2 等）|
| 输出形式 | 直接生成答案或隐状态 | 思维链推理过程 + 结构化偏好判断 |
| 训练方式 | 监督 / 自回归预训练 | RL (GRPO) + Rule-Based Reward |
| 典型用途 | 通用语言预训练 | RLHF / DPO 对齐、模型评估、Best-of-N 采样 |
| 能否互换？ | ❌ HRM 不能直接当奖励模型用 | ❌ RRM-A 不是新的基础架构 |

### 6.2 集成价值

- **LatentMind 暂无偏好对齐需求**（v1.0/v1.5 都是基座训练）
- 若未来需要 RLHF / DPO 阶段，RRM-A 是参考方向之一
- **不直接复用**——RRM-A 是训练范式而非网络层，不能嵌入 LatentMind 的 backbone

---

## 7. 相关参考

- [hrm-text.md](./hrm-text.md) — LatentMind 主干架构
- [trm.md](./trm.md) — RRM-B 谱系中的 TRM（小型递归基线）
- [hrm-original.md](./hrm-original.md) — RRM-B 谱系中的原始 HRM 27M
- [rrm-survey.md](./rrm-survey.md) — RRM-B 谱系完整调查（HRM, TRM, SE-RRM, Looped Transformer 等）
- [projects.md §2.5](./projects.md) — minimind 项目卡片（潜在对齐需求载体）

---

## 8. 关键引用块

> 摘要核心：
> "We introduce Reward Reasoning Model (RRM), a new reward modeling approach that reformulates scalar reward prediction as a chain-of-thought reasoning task, enabling adaptive test-time compute and improved interpretability."

> ⚠️ **声明**：以上引文为概念性引用，具体文字需从 arXiv 2505.14674 摘要验证。

---

**最后更新**：2026-06-13
**信息源**：arXiv 2505.14674 摘要 + 作者列表核查（[arxiv.org/abs/2505.14674](https://arxiv.org/abs/2505.14674)）+ RRM-A 与 RRM-B 区分整理
