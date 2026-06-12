# HRM（原始 27M）参考资料

> **论文**：*Hierarchical Reasoning Model*
> **作者**：Guan Wang, Jin Li, Yuhao Sun, Xingyi Yang, Chenyu Wang, Hongzhi Huang, Erjin Xie, Jing Shao
> **机构**：Sapient Intelligence
> **arXiv**：2506.21734（2025-06-26）
> **项目仓库**：https://github.com/sapientinc/HRM
> **本地源**：[`papers/arxiv-2506.21734-hrm-original.pdf`](./papers/arxiv-2506.21734-hrm-original.pdf) (2.0 MB)

---

## 1. 一句话定位

**脑启发的双时间尺度循环架构**：两个互相依赖的循环模块（高层慢规划 + 低层快计算），**27M 参数 + 1000 训练样本**就在 Sudoku、迷宫、ARC-AGI 上达到近完美。这是 HRM-Text（1.15B 文本版）的前身。

---

## 2. 摘要原文翻译

> 抽象推理（devising and executing complex goal-oriented action sequences）仍是 AI 的核心挑战。当前 LLM 主要用 CoT，但 CoT 任务分解脆弱、数据需求大、延迟高。**借鉴人脑分层与多时间尺度处理**，我们提出 **HRM**：一种新型循环架构，在保持训练稳定与效率的同时获得显著的计算深度。HRM 在**单次前向传播**中执行序列推理任务，**无需对中间过程做显式监督**，通过两个互相依赖的循环模块：一个**高层模块**负责**慢速抽象规划**，一个**低层模块**负责**快速细节计算**。**仅 27M 参数、1000 训练样本**，HRM 在复杂推理任务上取得卓越性能。模型**无需预训练或 CoT 数据**，却在复杂数独、最优迷宫路径、ARC 基准上达到近乎完美。

---

## 3. 核心创新

### 3.1 双模块互相依赖循环

```
高层 z_H（慢）: 全局规划、抽象任务分解
低层 z_L（快）: 局部细节、计算收敛
互相依赖: z_H → z_L（高层给低层上下文）→ z_H（低层结果反哺高层）
```

### 3.2 单次前向推理（无显式中间监督）

- 与 CoT 不同：CoT 需要模型生成中间 token
- HRM 整个推理在**隐藏状态**中完成，输出端只有最终答案
- 无需对中间状态做监督

### 3.3 借鉴大脑功能层级

- 前额叶：抽象规划、慢时间尺度
- 感觉运动皮层：快速细节、细粒度操作
- 两者形成"思考-行动"循环

### 3.4 不动点假设

- HRM 假设递归会**收敛到不动点**：z_L 和 z_H 在足够多步后趋近稳定
- 由此可用**1 步梯度近似**（Bai et al. 2019, Deep Equilibrium Models）：只对最后一步反向传播
- 这是 HRM-Text 中"预热 BPTT"的前身

### 3.5 极小数据 + 极小模型

- 27M 参数（比 GPT-2 Small 125M 还小 4×）
- 1000 训练样本（相比 LLM 数十亿 token）
- 0 预训练 → 完全从零开始
- 0 CoT 数据 → 隐藏状态推理而非显式中间步骤

---

## 4. 实验结果

| 任务 | HRM 27M | 之前最佳（小模型） | 备注 |
|---|---|---|---|
| 数独（极端难度） | 近 100% | 远低 | 之前小模型完全解不了 |
| 迷宫（大型）| 近 100% | 远低 | 路径规划任务 |
| ARC-AGI | 显著超越大 LLM | — | **关键 AGI 基准** |

论文 Table 1 显示：HRM 在 ARC-AGI 上超过 8B 规模、longer context 的 LLM。

---

## 5. 与 HRM-Text 的关系

| 维度 | HRM（27M） | HRM-Text（1.15B） |
|---|---|---|
| **领域** | 符号推理（Sudoku/Maze/ARC）| 自然语言 |
| **数据** | 1000 样本 | 40B unique tokens |
| **训练** | 从零 | 从零 |
| **成本** | 几小时单 GPU | $1,472 / 46h / 16 H100 |
| **循环结构** | H/L 互相依赖 | H/L 互相依赖（继承）|
| **归一化** | 标准 Norm | **MagicNorm**（增强）|
| **梯度** | 1 步近似（DEQ）| **预热 BPTT K=2→5** |
| **注意力** | 无（grid 嵌入）| **PrefixLM** |
| **目标** | 任务完成 | **任务完成 + PrefixLM** |
| **魔法点** | 不用显式监督 | 不用 CoT、不用预训练数据 |

**演进路径**：
- HRM 验证了"分层循环 + 不动点 + 单前向"的概念可行性
- HRM-Text 把这套范式扩展到语言，并解决了深层循环稳定性（MagicNorm）+ 训练效率（BPTT 预热）

---

## 6. 复现价值

### 6.1 对 LatentMind 的意义

- **概念根源**：MagicNorm 不是凭空发明的，源自原始 HRM 对"深层循环稳定性"的需求
- **架构模板**：H/L 双栈互相依赖的设计被 HRM-Text 直接继承
- **训练哲学**："单次前向 + 隐藏状态推理"在 v1.5（HRM + GRAM）仍然保留

### 6.2 不直接复用的部分

- 符号任务的数据格式（grid → token）不适合自然语言
- 27M 规模 vs 我们目标的 <1B 激活，扩展路径由 HRM-Text 验证
- 不动点假设在语言任务中**未必成立**——HRM-Text 改用 BPTT 预热就是承认这点

### 6.3 GitHub 仓库复用

- https://github.com/sapientinc/HRM
- 论文的官方实现，含 Sudoku/Maze/ARC 任务的数据 + 训练脚本
- 与 HRM-Text 仓库**姊妹**，可对比同一团队如何把同一架构迁移到不同领域

---

## 7. 关键引用块

> "Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency."

> "With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data."

> "These results underscore HRM's potential as a transformative advancement toward universal computation."

---

## 8. 与同领域其他工作关系

| 工作 | 关系 | 区别 |
|---|---|---|
| **Deep Equilibrium Models (DEQ)** | 理论基础 | DEQ 是不动点通用框架；HRM 是具体实现 |
| **Universal Transformer** | 循环 Transformer 先驱 | UT 共享参数但无分层；HRM 分 H/L |
| **Universal Transformer + adaptive depth** | 同方向 | UT 自适应深度；HRM 固定双层 |
| **Ouro / Huginn** | 后来的循环 LM | HRM-Text（1B）→ Ouro（1.4B）/Huginn（3.5B） |
| **TRM** | 直接简化版（见 [trm.md](./trm.md)） | TRM 用**单网络** 2 层达到 ARC-AGI 87% |

---

**最后更新**：2026-06-13
**信息源**：arXiv 2506.21734 全文、HRM 官方仓库、HRM-Text 论文引用
