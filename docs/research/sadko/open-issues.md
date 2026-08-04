# SADKO 文档体系审查：冲突、遗漏与待讨论清单

> **一句话定位**：对 docs/research 全部 8 份 SADKO 文档的一致性审查结果——技术冲突（需裁决）、数值/维度错误（需修正）、表述不一致（需统一）、设计遗漏（需补全）、开放问题（需讨论）
> **审查范围**：whitepaper · 64m-validation-plan · v1/v2/v3-architecture · hippo-vs-gdm-review · hippo-graph-emergence · hippo-phase0-manual · hippo-lifecycle
> **最后更新**：2026-07-29
> **状态**：大项已裁决（见 §7 落实记录），剩余 D/E 类问题（设计遗漏、开放讨论）仍待逐项推进。

**优先级说明**：🔴 阻塞实现（写代码前必须解决）｜🟡 影响验收（实验设计前必须解决）｜🟢 可暂缓（300M 前解决即可）

---

## 一、 A 类：技术冲突（需裁决）

### A-01 🔴 RoPE base 分配与 NTK 缩放理论倒挂

- **位置**：[whitepaper](./whitepaper.md) §2.2/§3.2 · [v1-arch](./v1-architecture.md) §3 决策表/§5.1 · [validation-plan](./64m-validation-plan.md) §2.3
- **问题**：所有文档称 "Static 低频 base=10k 保长程，Dynamic 高频 base=500k 保时序"。但按 RoPE 频率公式 $\theta_i = b^{-2i/d}$ 与 NTK 上下文外推实践（LLaMA 系长上下文都是**调大** base，10k→500k）：**base=10k 是高频快旋转（短程锐利），base=500k 才是低频长程配置**。即 base 分配与"保长程/保时序"的理由恰好倒挂；"低频/高频"的措辞也标反了。
- **连带影响**：v1.0 熵分离实验（Static 熵 < Dynamic 熵 且差值 >0.3）的理论预期需要重新推导——慢旋转头注意力更平坦、熵更高，当前判定标准可能因错误的理由而"意外正确"，也可能方向完全相反。
- **待裁决**：(a) 交换 base 分配（Static=500k 长程，Dynamic=10k 短程）；(b) 维持分配但修正理由与实验预期；(c) 先做一个 1-epoch 的 RoPE 消融预实验让数据裁决。**建议在写 v1.0 代码前完成 (c)。**

### A-02 🔴 压缩数据源层选择与融合方式不统一，且维度对不上

- **位置**：[validation-plan](./64m-validation-plan.md) §3.2/§3.3（"融合 L3+L5 KV"）· [v2-arch](./v2-architecture.md) §2 决策表（"L2/L3 + L4/L5"）· [v3-arch](./v3-architecture.md) §2（Hippo 输入"完整 chunk 的 L3+L5 KV 64×768"）
- **问题**：
  1. 层选择三个版本：L3+L5 / L2+L4（memory_write 代码取 anchor_layers[:2]）/ L2/L3+L4/L5；
  2. 单层 KV 每 chunk 展平维度为 64×(8 heads×48)=**64×384**。两层**均值池化**得 384 维，但 Hippo `kv_input_dim=768` 与 MLP Compressor 输入 768 都对不上；只有**拼接**两层才得 768。
- **待裁决**：统一为「L3+L5 **拼接**（64×768）」或改压缩器输入为 384；并同步修改 v2/v3 所有相关配置与代码。

### A-03 🔴 Cross-Attention 读取范围：先 Top-64 检索 vs 全量注意力

- **位置**：[validation-plan](./64m-validation-plan.md) §3.3（Top-K=64 chunks）· [v2-arch](./v2-architecture.md) §4.3（`CrossAttentionReadHead` 对 MemPool **全部 N 条**做注意力，无 Top-K 选择）· [v3-arch](./v3-architecture.md) §4.3（Router 返回 top_indices 但读取头未消费）
- **问题**：设计意图是"检索 Top-64 → CA 只读这 64 条"，但 v2 代码对全部存储（最多 1024 chunks）做注意力，检索环节被旁路。两者计算量与语义都不同。
- **待裁决**：确认"Router 检索 → CA 仅读取选中 chunks"为唯一数据流，并修正 v2/v3 读取头接口（`forward(hidden, mem_k_topk, mem_v_topk)`）。

### A-04 🟡 Latent MoE 在 64M 验证范围内的缺失

- **位置**：[whitepaper](./whitepaper.md) §2/§3.3（Latent MoE 为四大组件之一）vs 全部 64M 文档（dense 架构，无 MoE，四大实验也不覆盖路由）
- **问题**：白皮书将 Latent MoE（语义流形路由）列为核心壁垒，但 64M 三阶段完全不验证它。若 64M《已验证机制清单》不含 Latent MoE，则 300M 引入时无证伪依据，违背"64M 清单是唯一宪法"的原则。
- **待裁决**：(a) 明确 Latent MoE 推迟到 300M 并在白皮书中降级为"300M 待验证机制"；(b) 在 64M v3.0 增加第五个实验（Latent Router 原型）；(c) 承认 64M 清单只覆盖"记忆通路"，另立 300M 验证清单覆盖"路由通路"。

### A-05 🟡 "推理期零开销"的适用范围需要澄清

- **位置**：[whitepaper](./whitepaper.md) §2/§3.5（"推理期零开销 Zero-Overhead"）vs [v2-arch](./v2-architecture.md)/[v3-arch](./v3-architecture.md)（推理时 CA 持续读取 MemPool + 新内容需 Hippo 编码压缩）
- **问题**："零开销"仅指**扩散对齐的 Teacher 在推理时被丢弃**，但字面易被读为"右脑推理期免费"。实际推理期仍有：Hippo 编码新内容、FSQ 量化、Router 检索、CA 读取的开销。
- **待裁决**：在白皮书中将表述精确化为"对齐桥梁推理期零开销；右脑读写路径推理期开销为 O(N_mem) 常数级"，避免后续文档误引。

---

## 二、 B 类：数值/维度错误（需修正）

### B-01 🔴 Dual-Path FFN 参数量低估约 12 倍

- **位置**：[validation-plan](./64m-validation-plan.md) §2.2（"总新增参数 ~1.2M"）· [v1-arch](./v1-architecture.md) §3（"新增 ~1.2M，总参数 ~65.2M"）
- **核算**：单个 SwiGLU FFN（768→2048→768）= 3×768×2048 ≈ **4.72M**。每个锚点层新增 reason_ffn（memory_ffn 继承原始权重不计新增）= +4.72M，3 个锚点层 = **+14.2M**；加门控网络（~0.45M）与 CA 骨架（~0.89M），v1.0 实际新增 ≈ **15.5M，总参数 ≈ 79.5M**，而非 65.2M。
- **影响**："64M 小尺寸强约束"的论证基础、显存预算、以及与 MiniMind3 的 PPL 公平对比全部受影响。
- **待修正/裁决**：(a) 接受 ~80M 总量并更新所有文档；(b) 将 reason 通路 intermediate_size 降为 512-1024（新增降至 ~4-6M）；(c) reason_ffn 与 memory_ffn 共享 down_proj。**此决策同时是 E-02 的开放讨论。**

### B-02 🔴 memory_embed(16) 被用于索引 256 个 FSQ 码字

- **位置**：[v1-arch](./v1-architecture.md) §5.5（`memory_embed = Embedding(16, 768)`，语义标签用）· [v3-arch](./v3-architecture.md) §4.6/§5（`memory_kv = ar_student.memory_embed(codes)`，codes ∈ [0, 256)）
- **问题**：v3 直接用 v1 的 16 条目嵌入表索引 0-255 的 FSQ 码字，**越界**。且"语义标签嵌入"与"FSQ 码字→KV 嵌入"是两个不同功能，不应共用模块。
- **待修正**：v3 新增独立 `code_embed = Embedding(256, 768)`（或 codes → codebook_embeddings → 投影到 768），并在 v1/v3 文档中区分两个模块。

### B-03 🔴 v2 memory_write 跨层池化的张量维度错误

- **位置**：[v2-arch](./v2-architecture.md) §4.4
- **问题**：`k_layer.reshape(B, -1).mean(dim=0)` 得到的是 4×64×48=**12288 维**向量（且对 batch 维求均值），并非压缩机期望的 768 维。正确做法应是先对 chunk 内 token 维池化再按 head 拼接/投影：如 `k_layer.mean(dim=2).reshape(B, n_kv*head_dim)` → [B, 384]，两层拼接 → [B, 768]（与 A-02 联动）。
- **待修正**：与 A-02 一并重写该函数的张量流。

### B-04 🔴 hot_window=4096 与 max_position_embeddings=4096 撞车

- **位置**：[v1-arch](./v1-architecture.md) §2.1（max_position=4096）· [v2-arch](./v2-architecture.md) §3（hot_window=4096，训练 max_length=8192）
- **问题**：v2.0 要在 8K-16K 序列上训练并验证"4096 之外的远程记忆"，但位置编码上限只有 4096——超长部分没有合法 position_id，RoPE 外推未做任何配置（NTK/YaRN 均未提及）。
- **待修正**：v2 起将 max_position_embeddings 提至 16384 并明确 RoPE 外推方案；否则"远程记忆"实验在 64M 上根本无法构造。

### B-05 🟡 FSQ levels=[8,8,8,8] 与 codebook_size=256 不匹配

- **位置**：[validation-plan](./64m-validation-plan.md) §4.2（levels=[8,8,8,8]，codebook=256）· [v3-arch](./v3-architecture.md) §3（主推 [8,8,8,8]，备选 [8,8,4]）
- **问题**：[8,8,8,8] 组合空间为 4096，再 clamp 到 256 会浪费 94% 的组合结构且 clamp 制造大量同码冲突；[8,8,4] 恰好 =256 无浪费。两版并存但从未裁决。
- **待裁决**：64M 统一为 **[8,8,4]**（或 codebook_size 改为 4096 并重新评估码本利用率目标）。

### B-06 🟡 "压缩比"三套数字三种口径

- **位置**：[v2-arch](./v2-architecture.md) §2（"768/4，压缩比 4:1"，**维度压缩**）· [whitepaper](./whitepaper.md) §6.2（"300M 约 15:1，7B 后约 8:1"，口径未注明）· [graph-emergence](./hippo-graph-emergence.md) §四（"L/M ≈ 32x"，**token 数压缩**）
- **待修正**：增加术语定义表，区分「维度压缩比（768→192 = 4:1）」「token 压缩比（L/M）」「存储压缩比（含 K/V 双份与层数）」，并回填白皮书的 15:1/8:1 口径。

### B-07 🟢 Hippo 内部维度 192 vs 384

- **位置**：[v3-arch](./v3-architecture.md) §3（yaml `encoder_dim: 192` vs dataclass `hippo_hidden_size: 384`）
- **待修正**：二选一（影响 Hippo 参数量 ~5M 的核算与 input_proj 形状），建议 384（与早期设计一致，容量更足）。

### B-08 🟢 零散数值项

| 项 | 位置 | 不一致 |
| :--- | :--- | :--- |
| v3 总参数 | validation-plan（~70M）vs v3-arch（~72M/72.5M） | 以 B-01 修正后重算为准 |
| CA 骨架规格 | v1-arch yaml（6 heads×32=192）vs dataclass（8 heads×48=384） | dataclass 标注"等价形式"实为冲突，删除一版 |
| rms_norm_eps | 1e-5（yaml）vs 1e-6（dataclass） | 对齐 MiniMind3 原值 |
| 阶段 2b epochs | 2（plan/文本）vs 3（v2 yaml） | 统一 |

### B-09 🟠 Compressive 1D Conv 作为 FM 压缩的端侧备选（2026-07-31 新增）

- **位置**：[validation-plan §2.4](./64m-validation-plan.md)（FM 压缩配置）+ [v3-arch §3](./v3-architecture.md)（Hippo 主压缩方案）
- **来源**：[Compressive Transformers 论文笔记 §3.2](../../references/compressive-transformers.md)——1D Conv 压缩函数 f_c（stride=c, kernel=k≥c）
- **问题**：
  - 当前 Hippo 唯一压缩方案是 **Flow Matching（FM）**——端到端可微但计算成本较高
  - 64M 强约束下，若 FM 端侧延迟超预算，**无 Plan B**——风险集中
- **观察**：
  - Compressive Transformers 在 Enwik8/WikiText-103 上用 1D Conv（c=3 或 4）达到 SOTA
  - 1D Conv 是简单的"局部线性组合"压缩，端侧成本远低于 FM
  - 缺点：1D Conv 是**局部压缩**，远距离语义可能丢失
- **建议**：
  - 64M 阶段**先**验证 FM 压缩（[validation-plan 实验 A](./64m-validation-plan.md)）
  - 若 FM 端侧成本超预算（5ms 内难达），**回退评估** 1D Conv 作为 Plan B
  - 实验对比：(A) FM 压缩 (B) 1D Conv 压缩 (C) 无压缩（dense memory）—— 端侧延迟 + 重建损失 + 检索 Recall 三项指标
- **联动**：
  - 与 B-05 [8,8,4] SOP 一并裁决：FSQ 配置 + 压缩函数选型
  - 与 [hippo/retrieval-extraction.md §1](../../hippo/retrieval-extraction.md) 的"端侧备份方案"对应
- **关联论文**：[Memorizing Transformers](../../references/memorizing-transformers.md)（kNN 检索路径，FM 失败时的另一个 fallback）

---

## 三、 C 类：跨文档表述不一致（需统一）

### C-01 🟡 实验 A「内容寻址」通过阈值：10% vs 20%

- [validation-plan](./64m-validation-plan.md) §4.4 与 [v3-arch](./v3-architecture.md) §6 表格：通过 = 模糊 Recall 提升 **>20%**；[v3-arch](./v3-architecture.md) §6.1 代码：`passed = fuzzy_improvement > 0.10`。**同文档内代码与表格打架。**
- 待统一：建议 通过>20% / 证伪<10% / 中间为灰色区重调。

### C-02 🟡 实验 D「左脑校验」通过阈值：50% vs 70%，误杀 10% vs 20%

- 表格：过滤率 >70%、误杀 <10%；代码：`filter_rate > 0.50 and false_positive < 0.20`。
- 待统一：建议 通过>70% & 误杀<10% / 证伪<50% 或误杀>20%。

### C-03 🟡 Phase 3 vs Phase 3.5 命名与 EM 80% vs 95%

- [validation-plan](./64m-validation-plan.md) §4.3 称机械对齐为 "Phase 3"（验收 EM>80%），§4.6 决策树却称 "Phase 3.5 基线 EM>95%"；[whitepaper](./whitepaper.md) §10 执行清单又称 "Phase 3.5-Rote 基线 EM>95% 后再开始右脑验证"（时序与 plan 相反）。
- 待统一：建议「Phase 3 = FSQ 离散化；Phase 3.5 = 机械对齐；训练验收 EM>80%，实验启动基线 EM>95%（Rote 测试集）」，并修正白皮书执行清单的时序表述。

### C-04 🟡 Cross-Attention 骨架位置图示不一致

- [validation-plan](./64m-validation-plan.md) §2.2 架构图把 CA 骨架画在**锚点层（L2/4/6）内部**；[v1-arch](./v1-architecture.md) §3 明确 `cross_attn_layers: [3, 5]`。
- 待统一：以 **L3/L5** 为准（v2/v3 均按此实现），修正 validation-plan 架构图。

### C-05 🟡 Memory Embedding 形态：线性投影 vs 嵌入表

- [validation-plan](./64m-validation-plan.md) §2.2（"Memory Embedding: 768→768 零初始化"）vs [v1-arch](./v1-architecture.md) §5.5（`Embedding(16, 768)` 语义标签表）。与 B-02 联动，待统一为：语义标签表（16×768）+ v3 独立码字嵌入（256×768）。

### C-06 🟢 v1.0 PPL 验收线 2.75 vs 2.80

- [validation-plan](./64m-validation-plan.md) §2.5（≤2.75）vs [v1-arch](./v1-architecture.md) §8（≤2.75~2.80）与 §9 通过条件（≤2.80）。建议统一 ≤2.75，2.80 作为"延长训练"触发的灰色区。

### C-07 🟢 FSQ「量化维度」术语冲突

- [v3-arch](./v3-architecture.md)：4 个标量量化维度（levels）vs [graph-emergence](./hippo-graph-emergence.md) §1.3："量化维度 D ≈ log₂(N)+R，建议 64/128 起步"——两者说的不是同一个量（前者是 FSQ 标量数，后者像 latent 维/码本规模）。需统一术语并消除 64/128 与 256 码本的表面矛盾。

### C-08 🟢 whitepaper Scaling 表缺 64M 行

- [whitepaper](./whitepaper.md) §6.1 演进表从 300M 起步，但 64M 已有确定配置（Hippo 5M、FSQ 256、CA 标量门控 init=-2、全量 KL 之前的 Phase 对齐）。建议补一行 64M，使"64M→300M→1.5B→7B→70B"链条完整。

---

## 四、 D 类：设计遗漏（需补全）

| # | 优先级 | 遗漏项 | 说明 | 建议落点 |
| :--- | :--- | :--- | :--- | :--- |
| D-01 | 🔴 | **Hippo decode 模块未定义** | Phase 1 重构 Loss 需要 z→KV 的解码器（代码注释自承"需要实现 decode"），FM Decoder 的结构、ODE 积分方向、输出头（192→64×768?）全部缺失 | v3-arch §4.1 补 `decode()` 与 ODE solver 规格 |
| D-02 | 🟡 | **左脑校验模块无训练目标** | Verifier 的 threshold 标称"可学习"，但任何文档都未给出它的 loss（什么信号告诉它何时该触发二次检索？） | v3-arch §4.5 补训练目标（如对"触发后 EM 提升"做 RL/对比损失，或简化为固定阈值超参） |
| D-03 | 🟡 | **semantic_tag（16 维）来源未定义** | 动态门控与 Dual-Path FFN 都消费 semantic_tag，推理时填零向量则门控退化为静态——但"背诵 vs 推理"的门控分化正是实验 C 的验证对象。标签由谁产生（数据标注？左脑自产？）没有答案 | validation-plan §4.2 补标签生成方案；实验 C 设计需联动修正 |
| D-04 | 🟡 | **Memory Token 的 tokenizer 处理** | plan §0 自己提出"vocab=6400 小词表，Memory Token 需特殊处理"后再无下文 | v1-arch 补一节：是否扩充词表 / 用现有特殊 token / 完全走嵌入侧绕过 tokenizer |
| D-05 | 🟡 | **MemPool 的持久化与跨会话生命周期** | 64M 设计中 MemPool 是运行时 FIFO 缓冲区（max 1024 chunks 淘汰）；白皮书终极形态要求右脑"永不遗忘"的持久记忆。两者之间的桥（压缩产物如何落盘、检索、再加载）完全缺失 | whitepaper §7 或新增小节，明确 64M 范围外的演进路径 |
| D-06 | 🟡 | **训练/评估数据集构建方案** | kb_corpus、long_corpus_8k_16k、remote_recall_qa_50k、四大实验的 fuzzy/exact/association/recite/reason/distractor 测试集均只有文件名 | validation-plan 补数据构建附录（来源、构造规则、规模、防泄漏） |
| D-07 | 🟡 | **两个关键验收指标无度量定义** | "语义理解提升 >8%"（对什么基准、什么指标？）与"码字插值语义合理率 ≥60%"（谁评判、评分 rubric？） | v3-arch §5 补度量规格；phase0-manual 补插值评估 rubric |
| D-08 | 🟢 | **黄金测试集与关键注意力头的建立流程** | lifecycle 的熔断与 EWC 依赖"黄金标准测试集"和"Phase 0 标记的核心注意力头"，但建立方法未定义 | lifecycle §3 补建立流程（或注明随 phase0-manual 交付物产出） |
| D-09 | 🟢 | **事实感知三模块未落入 64M 架构** | hippo-vs-gdm §3.1 的 Fact-Aware Flow Weight / FSQ 弱监督锚定 / Fact-Gate 是正式决策，但 v3-arch 的模块与 Phase 流水线均未包含 | v3-arch 补入或在其中注明"推迟至 300M"并说明理由 |
| D-10 | 🟢 | **v2 用 hidden_states 冒充 KV 的简化** | v2-arch §5（旧稿 `forward_with_memory` 注释"简化： 用 hidden 代替 KV"）若进入实现，压缩的就不是设计要求的 Static KV | v2-arch 明确验收必须以真实 Static KV 为准，hidden 版仅限冒烟测试 |

---

## 五、 E 类：开放讨论问题（方向性）

| # | 问题 | 背景 | 建议讨论产出 |
| :--- | :--- | :--- | :--- |
| E-01 | **SADKO 与 LatentMind 主线（HRM-Text + GRAM）是什么关系？** | AGENTS.md 定义的 LatentMind 是 HRM 潜空间推理 + GRAM 多轨迹路线；SADKO 是"AR+FM 异构双脑"路线，两者哲学相近（压缩/潜空间）但架构不同。docs/research 未说明 SADKO 是替代、分支、还是借鉴对象 | 一段定位说明（写入 research/README），明确 SADKO 文档在本项目中的角色 |
| E-02 | **Dual-Path FFN 的参数预算怎么定？** | 见 B-01。接受 ~80M（参数量+24%）、缩小 reason 通路、还是共享部分投影？这决定"64M 强约束"的成色 | 裁决 + 回填 B-01 |
| E-03 | **图论验证工具链在 64M 阶段做到什么深度？** | Persistent Homology、谱分析、功能对齐测试的成本不低；64M 是"证伪"定位，是否只需 Attention Map 块状化 + Recon-结构耦合两个轻量信号，重分析留到 300M？ | 64M 图验证的最小集定义 |
| E-04 | **生命周期机制（稳定性分数、半衰期、码字均衡器）何时落地？** | lifecycle 文档是终态运维框架，64M 单轮训练用不上大部分机制；但黄金集与熔断在 64M 就有价值 | 标记哪些机制进入 64M、哪些推迟，并同步 phase0-manual |
| E-05 | **熵分离实验是否需要随 RoPE 裁决（A-01）重新设计？** | 若 base 分配或理由修正，实验 1.2 的预期方向与阈值（D−S>0.3）都要重推导 | 随 A-01 裁决一并更新 v1-arch §8.2 |
| E-06 | **"分层遗忘速率"与 FSQ 码本冻结是否冲突？** | lifecycle 允许 Memory $z$ 快速更新、码本冻结；但"深层范式近乎冻结"的载体是什么？若深层知识也住在 $z$ 里，冻结粒度只有"按 token 稳定性分数"一条路，这在工程上是否足够 | 明确各抽象层级知识的物理载体 |

---

## 六、 确认对齐良好的部分（无需改动）

以下为审查中确认跨文档一致、无需处理的设计：

1. **三阶段串行门禁**：v1.0→v2.0→v3.0 的启动许可证机制、每阶段"不可逾越"原则，在 plan、v1/v2/v3 架构文档间完全一致。
2. **消融隔离与强制早停**：LR 上限（主干 3e-4/右脑 1e-4）、每实验 2000 步、1000 步无改善终止、四实验同一数据集——plan 与 v3-arch §7 一致。
3. **门控初始化值**：CA 骨架 gate=-10（v1.0 零激活）→ -2.0（v2.0 激活）；Dual-Path gate=0.5 中性初始化——各文档一致。
4. **v2→v3 替换边界**：读取管线 100% 复用、仅替换压缩器（MLP→Hippo）与检索器（Cosine→FSQ Router）——plan §6 与 v3-arch 一致。
5. **负结果资产化哲学**：白皮书 §3.6/§4.2、plan §4.6、v3-arch §8、hippo 四份文档对"证伪清单与验证清单同等重要"的表述完全一致。
6. **图结构隐式涌现立场**：hippo-vs-gdm（放弃显式 GDM）、graph-emergence（禁止 $\mathcal{L}_{graph}$）、phase0-manual（结构探针只做观测不做损失）三者自洽。
7. **扩散对齐 Teacher-Student 设计**：0.3×KL+0.7×CE、2000-3000 步、Teacher 冻结后丢弃——白皮书、plan、v3-arch 一致。

---

## 七、 建议处理顺序

```text
第一波（写 v1.0 代码前）：
  A-01 RoPE 裁决（先做 1-epoch 消融预实验）
  B-01 参数预算裁决（联动 E-02）
  C-05 CA 位置统一（L3/L5）

第二波（写 v2.0 代码前）：
  A-02 + B-03 压缩数据源与张量流重写
  B-04 max_position 扩展方案
  A-03 检索-读取数据流确认

第三波（写 v3.0 代码前）：
  B-02 码字嵌入模块
  B-05 FSQ levels 裁决
  D-01 Hippo decode 规格
  D-03 semantic_tag 来源
  C-01/C-02/C-03 实验阈值统一

第四波（300M 规划前）：
  A-04 Latent MoE 定位
  D-05 MemPool 持久化
  E-01 SADKO 与 LatentMind 主线关系
  D-09 事实感知模块落位
```

---

## 八、 裁决落实记录（2026-07-29）

经用户裁决后，已在原文档中同步修改：

| 裁决项 | 裁决 | 落实位置 |
| :--- | :--- | :--- |
| **A-01 RoPE base 分配** | 交换 base：Static=500k（低频长程）、Dynamic=10k（高频短程），对齐 NTK 实践 | v1-arch §3/§5.1/§6/§8.2/§9、whitepaper §2.2、validation-plan §2.3、§2.5 |
| **A-02 压缩数据源层** | 改用 L3+L5 **拼接**（64×768 匹配 kv_input_dim），舍弃均值池化 | v2-arch §2/§3/§4.1/§4.4、v3-arch §2 |
| **B-01 参数预算** | 接受 ~80M 总量（诚实记录改造 ~15.5M），v1.0 总参数 ~80M | v1-arch §3/§6、whitepaper §6.1、validation-plan §2.2/§2.5 |
| **A-04 Latent MoE 64M 范围** | 64M 加第五实验（简化版）：2 experts + FSQ 码字 hash 路由 | v3-arch §6/§6.5/§6.6/§8/§9、whitepaper §2/§9/§10 |
| **E-01 SADKO 与主线关系** | 补充分支，与 LatentMind 主线（HRM-Text+GRAM）并行不替代；最终选择由两条路线各自 64M 清单决定 | research/README §0 |
| **C-01 PPL 验收线** | 统一为 ≤2.75 | v1-arch §8/§9 |
| **C-02 实验 A 阈值** | 统一为 通过>20% / 证伪<10% | v3-arch §6/§6.1/§8 |
| **C-03 实验 D 阈值** | 统一为 通过>70% 误杀<10% / 证伪<50% 或误杀>20% | v3-arch §6/§6.4/§8、whitepaper §4 |
| **C-04 Phase 3 vs 3.5 + EM** | Phase 3=FSQ, 3.5=机械对齐(训练 EM>80%), 启动门槛 Rote 测试集 EM>95% | v3-arch §8 |
| **B-04 max_position** | v2.0 起扩到 16384，NTK-aware 动态缩放（effective base = 40000） | v2-arch §2/§3 |
| **B-02 code_embed** | v3.0 新增独立 `FSQCodeEmbedding` 模块（256×768），与 v1.0 的 16 条 semantic_tag embedding 分离 | v3-arch §4.3 |

### 仍开放（未在本次裁决范围）

下列项仍需后续决策或留待 300M 阶段：

- **B-03 memory_write 张量流**：已通过 A-02 修复（拼接而非均值）
- **B-05 FSQ levels [8,8,8,8] vs [8,8,4]**：建议统一为 [8,8,4]=256，文档已更新但未明确"禁止 [8,8,8,8]"，等用户确认
- **B-07 压缩比术语**：建议统一表（维度/Token/存储三种口径），尚未补
- **A-03 检索-读取数据流**：尚未明确"Router Top-K → CA 仅读选中 chunks"接口规范
- **B-06 Hippo hidden 192 vs 384**：保留两版配置，待确认
- **C-07/B-08 零散**：CA 8×48 vs 6×32（dataclass 与 yaml 冲突）、rms_norm_eps、2b epochs 等小项
- **D 类**：Hippo decode 模块、D-02 校验训练目标、D-03 semantic_tag 来源、D-04 Memory Token tokenizer、D-05 MemPool 持久化、D-06 数据集构建、D-07 指标定义、D-08 黄金集建立、D-09 事实感知模块落位、D-10 v2 hidden 简化说明——共 10 项仍待补
- **E-02**（参预算二次确认）、**E-03**（图论验证深度）、**E-04**（生命周期机制落地阶段）、**E-05**（熵分离实验是否随 RoPE 重做）、**E-06**（多层遗忘速率粒度）
