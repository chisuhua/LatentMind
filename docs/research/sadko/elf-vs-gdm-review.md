# SADKO 右脑 vs GDM-v1.2：架构裁决与修订方案

> **一句话定位**：右脑技术路线的决策记录——为什么 Flow Matching + 双向注意力全面优于 Graph Diffusion，以及放弃显式 GDM 后的修订方案
> **关联文档**：[whitepaper.md](./whitepaper.md) §2.1（右脑 ELF 定位）· [elf-graph-emergence.md](./elf-graph-emergence.md)（图结构涌现机制）
> **最后更新**：2026-07-29

---

## 一、 背景

讨论源于对 SADKO（左脑 AR + 右脑 ELF/Flow Matching）架构与另一候选方案 **GDM-v1.2（Graph Diffusion Memory，图扩散记忆）** 的对比审查。结论：**SADKO 的"流匹配 + 双向编码"在数学本质和工程可行性上全面优于"图扩散"**。GDM 方案虽然抓住了"图基因"，但在生成范式选择上陷入"为了图而图"的局部最优，忽略了记忆压缩与提取的底层物理约束。

---

## 二、 核心差异对比（六维裁决）

| 维度 | GDM-v1.2 方案 | SADKO 方案 | 裁决 |
| :--- | :--- | :--- | :--- |
| **生成范式** | Discrete/Continuous Diffusion (DDPM/Score) | **Flow Matching (ODE)** | ✅ **SADKO 胜**。FM 训练更稳、采样步数更少、与 LLM latent space 对齐更自然。Diffusion 在连续语义空间易出现模式崩塌。 |
| **结构先验** | **显式图归纳偏置**（Node/Edge Latents, Graph-Aware Noise） | **隐式双向全连接**（Bidirectional Attention） | ⚠️ **平局，但 SADKO 更安全**。GDM 强加图结构可能导致过拟合；SADKO 的双向注意力是"最小结构假设"，让图结构从数据中自发涌现，泛化性更强。 |
| **方向性设计** | 未明确区分编码/解码方向性 | **精确分离**：编码双向（全局压缩），解码可逆 ODE，接口 Cross-Attn | ✅ **SADKO 胜**。解决了"全局记忆 vs 因果生成"的根本矛盾。GDM 未考虑此问题，推理时必然遭遇信息截断。 |
| **离散化** | RVQ / VQ-VAE | **FSQ（有限标量量化）** | ✅ **SADKO 胜**。FSQ 无码本坍塌风险，梯度直通，与 Flow Matching 的连续流形更兼容。RVQ 在记忆场景下极易退化。 |
| **协作机制** | Gated Cross-Attention（Learnable Gate） | **Cross-Attention + 扩散对齐桥梁** | ✅ **SADKO 胜**。"扩散对齐"在训练期就解决了方向不兼容，而非仅靠推理时的 Gate 硬拼接。这是"内生协作"而非"外挂协作"。 |
| **哲学一致性** | "记忆是条件生成的子图" | **"左脑因果之箭 + 右脑流形之环"** | ✅ **SADKO 胜**。将认知科学（双过程理论）与数学形式（AR + FM）统一，为后续扩展（多模态、规划）提供统一框架。 |

### 关键反思

GDM 方案犯了一个典型错误：**把"知识图谱"当成了存储格式，而非认知能力**。SADKO 正确地指出：**图结构不是存进去的，而是双向注意力 + 流匹配在压缩过程中自然学出来的**。只要右脑能看到全局、能可逆变换，它学到的 latent manifold 本身就同构于知识图谱的拓扑结构。强行注入图偏置反而是画蛇添足。

---

## 三、 基于 SADKO 的 v1.2 修订方案

保留 SADKO 核心架构，补全**"事实分离验证"**和**"图结构涌现探测"**两个实验模块，确保右脑不仅是优秀的压缩器，更是合格的 Memory Store。

### 3.1 架构微调：注入"事实感知"信号

SADKO 原文档侧重通用压缩，需增加事实性约束以防止右脑过度压缩语法/风格信息：

| 模块 | 机制 | 说明 |
| :--- | :--- | :--- |
| **Fact-Aware Flow Target** | 在 Flow Matching 训练时，对事实性 token（实体、数值、时间）对应的 latent 区域施加**更高的速度场权重** | 迫使右脑优先保留事实信息的变换轨迹 |
| **FSQ 码字语义锚定** | 训练初期用少量标注数据对 FSQ 码字做弱监督聚类（如"人物实体码"、"时间码"） | 为图结构涌现提供初始锚点。**注意：仅是弱监督初始化，不强制固定语义** |
| **Cross-Attn 事实门控（Fact-Gate）** | 左脑读取右脑 Memory 时增加轻量级 Fact-Gate，输入为左脑当前 query 的事实性评分（探针或简单分类器给出） | 非事实 query 降低对 Memory 的依赖，避免干扰语言生成 |

### 3.2 实验方案修订：SADKO + 图基因验证

| Phase | 目标 | 关键指标 | 死亡线 |
| :--- | :--- | :--- | :--- |
| **0. FM 收敛验证** | 确认 Flow Matching + FSQ 稳定训练 | Velocity Loss 收敛 + FSQ 码本利用率 >90% | Loss 震荡或码本坍塌 → 检查 ODE solver / FSQ 维度 |
| **1. 事实分离验证** | 证明右脑承载事实，左脑承载语言 | Fact-QA F1 ↓≥30%（移除右脑）+ Gradient Attribution Ratio >0.6 | Fact-QA ↓<15% → 增强 Fact-Aware Flow Weight |
| **2. 图结构涌现** | 验证双向 latent 自发形成图拓扑 | Latent Probe（依存/共指）Acc >70% + Topo Consistency >0.8 | Probe Acc <50% → 增加双向注意力层数或序列长度 |
| **3. 协作优势** | SADKO > 扁平基线 & RAG | Multi-hop QA F1 +10% vs MLP-Mem，Latency <50% vs RAG | 无显著增益 → 检查扩散对齐桥梁是否生效 |
| **4. 可逆性验证** | 验证 ODE 正反向一致性 | Encode→Decode Reconstruction F1 >0.9 | Recon F1 <0.8 → 检查 ODE solver 精度 / 步数 |

---

## 四、 关于"图扩散模型结构"的最终回应

> **不要再使用 Graph Diffusion Model。**

SADKO 的 **Flow Matching + Bidirectional Attention** 已经是"图原生"的：

- **双向注意力 = 完全图的连续松弛**
- **Flow Matching 的速度场 = 图上消息传递的连续极限**
- **FSQ 离散码字 = 图节点的量化表示**

不需要额外的图神经网络或图扩散。正确做法是：**在 SADKO 训练好后，用图论工具去分析它的 latent space，而不是在训练前把图论塞进模型里。**

如果 Phase 2 验证失败（latent 无序），正确的做法是：

1. 增加右脑深度/宽度
2. 延长训练数据中的长程依赖样本比例
3. 引入弱结构正则化（如 latent 距离与原文依存距离的相关性损失）

**而非回退到显式 GDM。**

---

## 五、 总结与行动指令

| 项目 | 决策 |
| :--- | :--- |
| **基础架构** | ✅ 采用 SADKO（AR + FM + FSQ + Cross-Attn） |
| **图结构实现** | ✅ 隐式涌现（双向注意力 + 流匹配），❌ 放弃显式 GDM |
| **事实分离保障** | ✅ 增加 Fact-Aware Flow Weight + Fact-Gate |
| **验证标准** | ✅ 四阶段验证（收敛 → 分离 → 涌现 → 协作） |
| **下一步行动** | 1. 实现 SADKO 基础版 → 2. 加入事实感知模块 → 3. 运行 Phase 0+1 → 4. 根据结果决定是否进入 Phase 2 |

> **核心结论**：SADKO 的"箭与环"隐喻不仅优美，而且数学上正确——它解决了记忆系统最根本的方向性矛盾。当前任务不是寻找更好的架构，而是**忠实地实现 SADKO，并用严苛的实验验证它是否真的长出了"图基因"**。
>
> **最好的结构不是设计出来的，而是在正确的物理约束下生长出来的。SADKO 提供了正确的物理约束。现在，让它生长。**
