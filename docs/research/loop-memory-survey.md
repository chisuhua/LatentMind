# 循环 Transformer + 持久记忆系统：文献谱系

> **一句话定位**：梳理"循环 Transformer"与"持久记忆 KV 压缩"两条**正交**研究脉络，建立统一的三维分类法（intra-decode / inter-decode / prefill-decode），澄清 DiscoLoop 与 Hippo 等项目组件各自的归属与互补关系
> **性质**：跨研究方向调研（inform Logos / Hippo / Thumos 的未来决策）
> **最后更新**：2026-07-31
> **触发动机**：用户调研 DiscoLoop 时发现的概念混淆（"一个 decode 输出是一个循环"），需澄清三种循环维度
> **核心澄清**：用户描述的"实时压缩 KV → 连续记忆层 → 离散 Q 对齐"是 **inter-decode 记忆压缩**流派的扩展，**不是** DiscoLoop 的 **intra-decode 表征对齐**

---

## 0. 三种"循环"维度（必读，概念基础）

> ⚠️ **本文最重要的章节**：所有后续讨论都基于这里的定义。若跳过此节，下游理解会全部错位。

### 0.1 现代推理过程中存在的三种"循环"

```
                ┌──────────────────────────────────────────────────────────────┐
                │                    多轮交互（用户↔模型）                       │
                │                                                              │
                │   Round 1:      Round 2:        Round 3:        Round N:      │
                │   ┌────────┐    ┌────────┐      ┌────────┐      ┌────────┐    │
                │   │Prefill │    │Prefill │      │Prefill │      │Prefill │    │
                │   │Decode1 │    │Decode2 │      │Decode3 │      │DecodeN │    │
                │   │Decode2 │    │Decode3 │      │...     │      │...     │    │
                │   │...     │    │...     │      │DecodeN │      │DecodeM │    │
                │   │DecodeK │    │DecodeM │      │        │      │        │    │
                │   └────────┘    └────────┘      └────────┘      └────────┘    │
                │                                                              │
                │   ←── Loop C (Prefill↔Decode 交替) ──→                       │
                └──────────────────────────────────────────────────────────────┘
                            │
                            │ 单次 Round 内部放大：
                            ▼
                ┌──────────────────────────────────────────────────────────────┐
                │                单次 Decode（生成 1 个 token）                  │
                │                                                              │
                │   输入: KV_cache[t-1 tokens] + 当前 hidden state              │
                │                                                              │
                │   for k = 1, 2, ..., K:                                       │
                │       H^(k+1) = f_θ(H^(k))   ← 共享 KV cache + 共享权重      │
                │                                                              │
                │   输出: 第 t 个 token 的 logits                                │
                │                                                              │
                │   ←── Loop A (intra-decode 循环, K 次迭代) ──→               │
                └──────────────────────────────────────────────────────────────┘
                            │
                            │ 多次 Decode 之间的全局维护：
                            ▼
                ┌──────────────────────────────────────────────────────────────┐
                │            序列生成过程中的记忆维护（跨 Decode）                │
                │                                                              │
                │   KV_cache: [tok1, tok2, tok3, ..., tokN]   (持续增长)       │
                │   Memory:   [mem1, mem2, mem3, ..., memM]    (M << N)        │
                │                                                              │
                │   每生成 N 个 token:                                          │
                │       compress(KV_cache_window) → mem_new                    │
                │                                                              │
                │   每次 Decode 时:                                              │
                │       query(memory) + KV_cache_window                          │
                │                                                              │
                │   ←── Loop B (inter-decode 记忆压缩) ──→                      │
                └──────────────────────────────────────────────────────────────┘
```

### 0.2 维度对照表

| 维度 | Loop A: Intra-decode 循环 | Loop B: Inter-decode 记忆压缩 | Loop C: Prefill-Decode 交替 |
|------|--------------------------|------------------------------|---------------------------|
| **别名** | Recurrence depth / K-loop | Persistent memory | Multi-turn dialogue |
| **作用域** | 1 个 token 的生成 | N 个 token 的序列 | 整个对话 session |
| **迭代次数** | K（典型 2-8）| M（典型 100-1000）| 用户驱动 |
| **状态载体** | Hidden state H | KV cache + Memory pool | Conversation history |
| **KV cache 行为** | K 次迭代共享同一 KV cache | 每 token 增 1 行；每 N token 压缩 1 次 | 每轮整体重建 |
| **代表性架构** | HRM-Text, TRM, DiscoLoop, Logos H/L | Memorizing Transformer, Compressive Transformer, Hippo | ChatGPT / Claude / 系统提示重读 |
| **核心问题** | "如何在 K 次迭代内对齐隐状态？" | "如何在长序列中压缩 + 检索？" | "如何维护多轮一致性？" |
| **用户曾误以为** | "一个 decode = 一个循环" ← **错** | 用户描述的概念 = 这一类 | 通常无歧义 |
| **用户描述的 Q↔Memory 对齐** | ❌ 不属于这一类 | ✅ **属于这一类**（详见 §1）| ❌ |

### 0.3 用户的关键误解（必须先纠正）

> **误解**："我理解一个 decode 输出是一个循环"
>
> **正确**：
> - 在**标准 Transformer**（GPT-3 / Llama / Qwen）：一个 decode 输出 = **一次** forward pass = **零**循环（无迭代）
> - 在**循环 Transformer**（HRM / TRM / DiscoLoop / Logos）：一个 decode 输出 = **K 次** forward pass（共享权重、共享 KV cache），其中 K 是超参数
> - K=2 的 DiscoLoop = 生成 1 个 token = 2 次循环迭代
> - Logos 的 H/L 双时间尺度 = 生成 1 个 token = (2 H-cycles × 3 L-cycles) = **6 次**循环迭代

**隐含的推理**：
- K 次循环**不增加**序列长度（KV cache 不增长）
- K 次循环**改变**的是 hidden state 的精炼深度
- "一个 token 内 K 个循环"是**计算深度**维度，**不是**"序列长度"维度

---

## 1. 用户提出的核心概念精解

### 1.1 用户原话

> "实时压缩 KV 到连续记忆层，后续的离散查询 Q 和记忆层对齐"

### 1.2 这个概念属于哪一类？

**Loop B (inter-decode 记忆压缩)**，具体定位：

```
┌────────────────────────────────────────────────────────────┐
│  用户提出的概念（精确还原）                                  │
│                                                            │
│  ┌──────────────────────────────────────┐                  │
│  │  Step 1: 实时压缩                    │                  │
│  │                                      │                  │
│  │  KV_cache[N tokens]                  │                  │
│  │       ↓ (compress)                   │                  │
│  │  M (continuous memory layer)         │ ← "连续记忆层"   │
│  └──────────────────────────────────────┘                  │
│                    ↓                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Step 2: 离散查询                    │                  │
│  │                                      │                  │
│  │  Q = discrete_query_token            │ ← "离散查询 Q"   │
│  │       ↓ (embed to continuous)        │                  │
│  │  Q_emb                                │                  │
│  │       ↓ (attend over M)              │                  │
│  │  retrieved_content = Attention(Q_emb, M)               │
│  │       ↓                                │                  │
│  │  Q ↔ M 对齐结果（用于下一步推理）     │ ← "对齐"         │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  关键：这是 LOOP B 维度（跨 decode），不是 LOOP A           │
└────────────────────────────────────────────────────────────┘
```

### 1.3 与 DiscoLoop 的对比（核心澄清）

| 维度 | 用户概念 (Loop B 记忆压缩) | DiscoLoop (Loop A 表征对齐) |
|------|---------------------------|------------------------------|
| **时间跨度** | 跨 N 个 token（长程）| 单个 token 内 K 次迭代（短程）|
| **压缩** | ✅ 显式压缩 KV → memory | ❌ **不压缩**任何东西 |
| **存储** | 独立 memory 层（持久化）| 残差流（瞬时）|
| **查询 Q** | 离散 token → 嵌入 → attention | h^(k+1) → Φ(h) → 加到残差 |
| **"对齐"的方向** | Q_emb → attention(M) → 输出 | H → Φ(H) → 加回到 H |
| **本质问题** | 长上下文复用 | 短程分布对齐 |
| **典型用户** | 需要超长上下文的场景 | 需要复杂单步推理的场景 |

**最本质的区别**：
- DiscoLoop 的 `Φ(h) = Σ p_v(h) W[v]` **不是记忆检索**，而是**自解码再嵌入**——它从 h 自己解码出的分布采样嵌入向量，**没有查询任何外部存储**
- 用户概念的 Q → M **是真正的记忆检索**——Q 从外部 memory 中查询相关片段

### 1.4 "对齐"二字的精确语义

| 概念 | "对齐"的精确语义 | 涉及维度 |
|------|----------------|---------|
| **DiscoLoop 的对齐** | 把连续隐状态 h 的几何分布拉回 E(·) 锚定的训练分布（解决表征错位）| Loop A（intra-decode）|
| **用户描述的对齐** | 让查询 Q 与记忆 M 在语义空间相互对齐（解决检索召回率）| Loop B（inter-decode）|
| **项目 Hippo / SADKO 的对齐** | 让 AR（连续 logits）与 Hippo（离散 FSQ）通过扩散蒸馏对齐（解决跨模块分布错位）| Loop B（cross-module）|
| **Logos Nano-WM Gate 的对齐** | 让外部知识 KV 与 Logos hidden state 在门控信号上对齐（解决知识注入）| Loop B（cross-component）|

**四种"对齐"虽然都用同一个词，但解决完全不同维度的问题**。

---

### 1.5 DiscoLoop × PLT 对齐统一视角（2026-07-31 新增，跨流派发现）

> 📌 **本节为项目关键发现**：本节源自用户洞察——PLT 的后一轮输入 `shift(h^(r-1)_{i-1})` 与 DiscoLoop 的 H^(k) 面临**同一类问题**（h 偏离 E 分布），但**触发维度不同**。本节为这种"对齐"建立统一框架。

#### 1.5.1 两种"对齐"的精确定义

| 维度 | **DiscoLoop 对齐**（intra-loop / same-position）| **PLT 跨位置对齐**（inter-position / same-loop）|
|------|----------------------------------------------|--------------------------------------------------------|
| **错位的 h 来源** | 同位置前一轮的 H^(k) | 另一位置前一轮的 shift(h^(r-1)_{i-1}) |
| **错位的本质** | 同一 token 多次循环后**纵向漂移** | 不同 token 同一轮内**横向错位** |
| **信息流动方向** | 时间（K 轮迭代向下）| 空间（位置 i-1 → i 横向）|
| **触发点** | 任意 token 经过 K 次循环后 | 任意轮次的位置 i（依赖 i-1）|
| **f_θ 看到的输入** | H^(k) + E(x) | E(x_i) + shift(h^(r-1)_{i-1}) |
| **不对齐的根源** | 同一 token 的 hidden state 偏离嵌入分布 | 跨 token 的 hidden state 偏离嵌入分布 |
| **类比** | 同一**行**的"时间老化" | 同一**列**的"空间错位" |
| **类比图示** | `t=0 → t=1 → t=2`（纵向）| `pos=0 → pos=1 → pos=2`（横向）|
| **论文来源** | [discoloop.md §3.1](../references/discoloop.md) | [loopcoder-v2.md §3.1](../references/loopcoder-v2.md)（PLT 公式）|
| **解法 Φ 的输入** | H^(k) （同位置）| shift(h^(r-1)_{i-1}) （跨位置）|
| **Φ 输出** | 软嵌入（"h 的词表软期望"）| 软嵌入（同左）|
| **解法核心** | "把纵向漂移的 h 拉回 E 分布" | "把横向错位的 shift(h) 拉回 E 分布" |

#### 1.5.2 关键统一视角

**两者本质相同**：
- 共同点：f_θ 在"训练时消费 E"和"实际消费某 h"之间存在分布失配
- 共同解法：Φ 通道把 h 投影回 E 分布 → 恢复 f_θ 的"舒适区"

**两者触发点不同**：
- DiscoLoop：纵向（时间）漂移
- PLT：横向（空间）错位
- → Φ 的应用位置不同（输入侧 / 位置侧 / 输出侧）

**统一公式**（DiscoLoop + PLT + Φ 集成）：

```python
# 通用"分布对齐"算子（可应用于任何 h 错位场景）
def phi(h):
    """Soft decode-then-encode: 把任意 h 投影回 E 分布"""
    p = softmax(lm_head(h) / tau, dim=-1)   # h → 词表概率
    return einsum("btv,vd->btd", p, embedding_table)  # 词表概率 → 嵌入期望
```

#### 1.5.3 PLT + Φ 集成方式（三种）

**方式 A：Φ 在 shift 前（输入对齐）**

```python
def plt_phi_input(h_prev, x, r):
    h_shifted = shift_right(h_prev)              # [B, T, d]
    h_aligned = phi(h_shifted)                    # ⭐ 关键：先 realign 再 add
    B_r = x + h_aligned                            # f_θ 看到"两条都是 E 分布"
    h_r = transformer_block(B_r)
    return h_r
```

**方式 B：Φ 在 shift 后（位置对齐）**

```python
def plt_phi_position(h_prev, x, r):
    h_shifted = shift_right(h_prev)
    B_r = x + h_shifted
    h_r = transformer_block(B_r)
    h_r = h_r + alpha * rms_norm(phi(h_r))        # ⭐ 输出端 DiscoLoop 范式
    return h_r
```

**方式 C：双端对齐（输入 + 输出）**

```python
def plt_phi_both(h_prev, x, r):
    h_shifted = shift_right(h_prev)
    h_aligned = phi(h_shifted)                    # 输入端
    B_r = x + h_aligned
    h_r = transformer_block(B_r)
    h_r = h_r + alpha * rms_norm(phi(h_r))        # 输出端
    return h_r
```

#### 1.5.4 对 Logos H/L 架构的具体应用

```python
# Logos L-PLT + Φ 集成（结合 §4.6 L-PLT + §3.5 DiscoLoop 探针）
def l_plt_with_phi(h_L_prev, h_H, l):
    h_L_shifted = shift_right(h_L_prev)        # 跨位置 L 状态
    h_L_aligned = phi(h_L_shifted)             # ⭐ 对齐到 E 分布
    h_L_input = h_H + h_L_aligned
    h_L_new = L_module(h_L_input)
    return h_L_new
```

**对 Logos 64M 验证计划的影响**：
- §3.5（DiscoLoop 探针）若显示错位 → Φ 模块已就位
- §4.6（L-PLT 评估）若启用 → Φ 已在 h_H 路径中，可**直接**用方式 A
- 两节不再是独立实验，而是**共享 Φ 模块**的联合设计

#### 1.5.5 对项目决策的关键启示

1. **§3.5（Φ）与 §4.6（L-PLT）应联合验证**而非独立
2. **PLT + Φ 是"延迟 + 质量"联合优化**：
   - 单独 PLT：延迟 -33%，但 shift 错位可能损害质量
   - 单独 Φ：质量 +OOD，但无延迟改善
   - 联合 PLT + Φ：延迟 -33% + 质量稳定
3. **对项目其他研究的暗示**：
   - Hippo 胼胝体契约的"对齐"也是跨模块分布失配（属于"跨模块"维度）
   - Thumos 的"agent 内化"也是某种"内部对齐"（外部 → 内部）
   - 四种对齐（DiscoLoop / PLT / 跨模块 / 跨组件）可统一在一个"f_θ 消费分布失配"框架下

#### 1.5.6 决策原则

| 端侧约束 + 质量要求 | 推荐方案 |
|-------------------|---------|
| K=2 端侧预算充足 | ❌ 不引入任何 PLT/Φ（保持串行 H/L）|
| K=4 端侧预算紧张 | ✅ 仅 L-PLT（节省 33% 延迟，不引入 Φ）|
| K=4 + OOD 性能要求高 | ✅ L-PLT + Φ 方式 A（联合优化）|
| K=6-8 端侧仍超预算 | ✅ L-PLT + Φ；若仍超 → §2.6 Loop B fallback |
| K>8 | ❌ 不再叠加 PLT 变体；用 §2.6 StreamingLLM |

详见 [logos/k-strategy.md §2.2.7](../logos/k-strategy.md) 完整设计与 [logos/64m-validation-plan.md §4.6.9](../logos/64m-validation-plan.md) 联合实验设计。

---

## 2. 完整文献谱系

### 2.1 Loop A: Intra-decode 循环系（每个 token K 次迭代）

| 论文/架构 | 年份 | 核心机制 | 循环数 K | 记忆机制 | 对齐方式 |
|----------|------|---------|---------|---------|---------|
| **HRM-Text** (Sapient) | 2026-05 | 分层 H/L 双时间尺度 | 8 (2H × 3L) | 无 | 加法注入（无对齐通道）|
| **TRM** (Samsung SAIL) | 2025-10 | 单 block 递归 + adaptive computation | 3-16 (adaptive) | 无 | 残差连接 |
| **Huginn 3.5B** | 2025-02 | latent recurrent + CLP | 50 (并行) | 无 | 累积 hidden state |
| **LoopCoder-v2 / PLT** | 2026-06 | CLP + G-SWA 并行循环 | 1-4 (R=2 最优) | 无 | **位置偏移**打破串行 |
| **DiscoLoop** (UC Berkeley) | 2026-07 | 离散嵌入通道 + Φ soft decode | 2 (tested) | 无（瞬时）| **Φ 通道对齐**（用户概念核心问题）|
| **Logos H/L** (本项目) | 2026-07 | 分层 H/L + Split-GQA + Nano-WM | 6 (2H × 3L 默认) | Nano-WM 外部 KV | 加法注入 + Nano-WM Gate |

**共同特征**：
- K 次迭代共享 KV cache（KV 不增长）
- K 次迭代共享权重（参数不变）
- 每次迭代的 hidden state 越来越精炼
- **无持久记忆**：每生成完一个 token，hidden state 信息只保留到该 token 的输出

### 2.2 Loop B: Inter-decode 记忆压缩系（跨 token 记忆维护）

| 论文/架构 | 年份 | 压缩粒度 | 记忆形式 | 查询方式 | 检索粒度 |
|----------|------|---------|---------|---------|---------|
| **Compressive Transformer** (DeepMind) | 2020 | 每 512 tokens | 连续向量（粗粒度 + 细粒度双缓冲）| 双注意力 | segment |
| **Memorizing Transformers** (Meta) | 2024 | 实时（每个 segment）| 连续向量 + kNN 索引 | kNN 检索 | 单 token |
| **AutoCompressors** (Stanford) | 2023 | 模型自生成摘要向量 | 连续向量（prompt 嵌入）| 直接 prompt | summary |
| **∞-former / RMT** | 2022 | 持续压缩 | 连续记忆单元 | 注意力 | segment |
| **StreamingLLM** (MIT) | 2024 | 滑动窗口 + attention sink | 固定锚点 tokens | 注意力 | sliding |
| **InfLLM** | 2024 | 滑动 + 关键块保留 | 连续记忆层 | 注意力 | chunk |
| **Hippo** (本项目) | 2026-07 | Flow Matching 压缩 | FSQ 离散码本 | I3 Retrieval API | Top-K chunks |

**共同特征**：
- 长 KV cache 被周期性压缩为低维记忆
- 记忆可被新生成内容复用
- 查询 Q 通过注意力或检索与记忆交互
- **持久化**：记忆跨多个 token 保留

### 2.3 Loop C: Prefill-Decode 交替（多轮交互）

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| **标准多轮对话** | 每轮完整 prefill + decode | ChatGPT, Claude |
| **系统提示重读** | 每 N 轮重新 prefill system prompt | 长程 persona 维护 |
| **KV cache 跨轮复用** | 缓存上一轮的 prefill KV | 减少重复计算 |
| **Recursive summarization** | 每轮生成摘要，追加到下一轮 prefill | 长对话压缩 |

**共同特征**：
- 整轮作为最小单元
- 跨轮由用户消息或系统触发
- 与本文核心讨论（intra/inter-decode）正交

---

## 3. 用户概念的精确归属与扩展

### 3.1 用户描述的核心架构图

```
                  ┌────────────────────────────────────────────────┐
                  │        用户提出的"实时压缩 + 离散查询"           │
                  │                                                │
  输入 tokens ────┤  ┌──────────────────┐                          │
                  │  │   Prefill         │                          │
                  │  │   KV_cache 生成    │                          │
                  │  └─────────┬─────────┘                          │
                  │            ↓ (实时，每 N tokens)                │
                  │  ┌──────────────────┐                          │
                  │  │  Compressor      │                          │
                  │  │  (FM / Linear)   │                          │
                  │  └─────────┬─────────┘                          │
                  │            ↓                                    │
  ┌────────────┐  │  ┌──────────────────┐                         │
  │  Memory    │←─┼──│  Continuous M     │ ← 连续记忆层            │
  │  Pool      │  │  │  (vector store)   │                         │
  └─────┬──────┘  │  └──────────────────┘                         │
        ↑         │            ↑                                    │
        │         │  ┌──────────────────┐                          │
        │         │  │  Query Q (discrete)│                        │
  检索  │         │  │  - token id       │ ← 离散查询              │
        │         │  │  - code (FSQ)     │                          │
        │         │  │  - hidden state    │                          │
        │         │  └─────────┬─────────┘                          │
        │         │            ↓ (embed + attend)                  │
        └─────────┼────────────┘                                   │
                  │                                                │
                  └────────────────────────────────────────────────┘
```

### 3.2 用户概念与现有架构的对应

| 用户概念元素 | 对应现有架构 | 距离 |
|--------------|-------------|------|
| 实时压缩 KV → M | **Hippo FM 压缩** ([sadko/whitepaper §2.1](../research/sadko/whitepaper.md)) | ✅ 直接对应 |
| 连续记忆层 | **Hippo FSQ 码本**（虽然是离散，但通过 soft expectation 可视为连续）| ⚠️ 需澄清 |
| 离散查询 Q | **I3 Retrieval API 的查询向量** | ✅ 直接对应 |
| Q ↔ M 对齐 | **扩散对齐（Distillation）** | ✅ 训练时；⚠️ 推理时不同 |
| 实时性（每 N tokens）| **未明确**——Hippo 当前是离线压缩 + 在线检索 | ❌ 需扩展 |

### 3.3 用户概念的"扩展性"分析

**用户提出的方向是对的**，但相比当前 Hippo 增加了三个新维度：

1. **实时性**：从"离线压缩 + 在线检索"扩展到"推理时实时压缩"
2. **双向对齐**：Q ↔ M（不仅是 M → Q，而是双向）
3. **持续记忆层**：独立于 AR hidden state 的 memory pool

这些扩展 **对项目而言是有价值的研究方向**，但需要谨慎评估：

| 维度 | 现状 | 扩展方向 | 风险 |
|------|------|---------|------|
| 压缩时机 | 离线（训练时 FM 压缩）| 推理时实时压缩 | 端侧算力预算紧张 |
| 记忆形式 | FSQ 离散码本 | 连续向量 + 离散混合 | 显存占用增加 |
| 对齐方式 | 训练时扩散蒸馏 | 推理时双向 Q↔M | 增加推理延迟 |
| 与 H/L 循环关系 | 外部模块 | 嵌入到 H/L 循环 | 破坏现有架构 |

---

## 4. 对 Logos 主线的具体启示

### 4.1 短期（64M 阶段）

**专注 Loop A（intra-decode 表征对齐）**：
- ✅ §3.5 表征对齐探针（已实施）——验证 H/L 循环的 DiscoLoop 假设
- ❌ **不**引入 Loop B 记忆压缩——64M 强约束禁止复杂机制

### 4.2 中期（300M 阶段）

**视 §3.5 探针结果决定**：

| §3.5 探针结果 | 推荐行动 |
|--------------|---------|
| H/L 不存在显著错位 | 维持现状；可考虑 Loop B 集成 |
| H/L 存在显著错位 + Φ 通道修复成功 | 加入 Φ 通道；暂不集成 Loop B |
| H/L 存在显著错位 + Φ 通道无效 | 重新设计 H/L；探索 Loop B 作为替代方案 |

### 4.3 长期（1B+ 阶段）

**Loop A + Loop B 融合（待 §3.5 探针 + Hippo 验证后启动）**：

```
┌─────────────────────────────────────────────────────┐
│  Logos v3 (1B+) 设想                                  │
│                                                      │
│  输入 tokens                                          │
│      ↓                                               │
│  Prefill + KV cache 生成                              │
│      ↓                                               │
│  ┌──────────────────────────────────────┐             │
│  │  Loop B: 每 N tokens 压缩到 M         │  ← "实时压缩"│
│  │  Hippo FM 压缩 + FSQ 码本             │             │
│  └──────────────────────────────────────┘             │
│      ↓                                               │
│  ┌──────────────────────────────────────┐             │
│  │  Loop A: 每 token K=2-6 次循环        │             │
│  │  - H/L 双时间尺度                     │             │
│  │  - DiscoLoop 式 Φ 通道（若需要）       │             │
│  │  - Nano-WM Gate 消费 Hippo 检索       │             │
│  └──────────────────────────────────────┘             │
│      ↓                                               │
│  输出 token                                          │
└─────────────────────────────────────────────────────┘
```

**关键约束**：Loop A 和 Loop B 的集成必须**独立验证后再融合**（详见 [fsq-consumption-design.md](../research/logos/fsq-consumption-design.md)）。

---

## 5. 关键澄清与防误解检查清单

### 5.1 用户常见误解

| 误解 | 正确 |
|------|------|
| "一个 decode 输出是一个循环" | 一个 decode 输出 = K 次循环（K 是迭代深度）|
| "DiscoLoop 是记忆系统" | DiscoLoop 是 intra-decode 表征对齐，**无记忆** |
| "压缩和提取是同一回事" | 压缩（write，跨 N tokens）+ 提取（read，单次 query）|
| "连续记忆层和离散码本等价" | 不等价——连续可微分；离散是 FSQ 锚点 |
| "对齐就是相似度匹配" | 对齐有三种：分布对齐（DiscoLoop）/ 检索对齐（用户概念）/ 训练对齐（扩散蒸馏）|

### 5.2 决策检查清单（实施任何循环+记忆架构前）

- [ ] 我说的是 Loop A 还是 Loop B？（决定适用论文）
- [ ] KV cache 在 K 次循环中**共享**还是**复制**？（决定显存）
- [ ] 记忆压缩发生在**离线**还是**实时**？（决定端侧可行性）
- [ ] 查询 Q 是**离散**还是**连续**？（决定检索机制）
- [ ] "对齐"指的是**哪种**？（分布/检索/训练）

---

## 6. 相关文档

| 文档 | 关系 |
|------|------|
| [../references/discoloop.md](../../references/discoloop.md) | DiscoLoop 论文笔记（Loop A 代表）|
| [../references/loopcoder-v2.md](../../references/loopcoder-v2.md) | LoopCoder-v2 笔记（Loop A 并行化）|
| [../references/per-token-convergence.md](../../references/per-token-convergence.md) | 动态 K（Loop A 早退）|
| [../research/sadko/whitepaper.md](../sadko/whitepaper.md) | SADKO 异构双脑（Loop B 模块化）|
| [../research/hippo/README.md](../hippo/README.md) | Hippo 胼胝体契约（Loop B 接口）|
| [../research/logos/64m-validation-plan.md §3.5](../logos/64m-validation-plan.md) | 表征对齐探针（Loop A 实验）|
| [../research/logos/64m-validation-plan.md §3.6](../logos/64m-validation-plan.md) | Loop B 备选实验（条件启动，三档 fallback）|
| [../research/logos/64m-validation-plan.md §4.6.9](../logos/64m-validation-plan.md) | L-PLT + Φ 联合实验（延迟 + 质量联合优化）|
| [../research/logos/k-strategy.md §2.2.7](../logos/k-strategy.md) | PLT + Φ 集成设计（DiscoLoop 对齐 vs PLT 跨位置对齐）|
| [../research/logos/fsq-consumption-design.md](../logos/fsq-consumption-design.md) | Logos 消费 Hippo FSQ 条件设计（Loop A + Loop B 融合）|
| [../research/four-lines-synthesis.md](../four-lines-synthesis.md) | 四研究线综合脉络 |

---

## 7. 文献调研完成度（2026-07-31 更新）

> 本节追踪 Loop B 完整文献谱系的调研完成度。已完成笔记标 ✅，未完成标 ⏳。

### 7.1 已完成（Loop B 6 个核心流派全覆盖）

| # | 论文 | 年份 | 流派 | 笔记 | 状态 |
|---|------|------|------|------|------|
| 1 | **Memorizing Transformers** (Google) | 2022 | kNN 检索 | [memorizing-transformers.md](../references/memorizing-transformers.md) | ✅ |
| 2 | **Compressive Transformers** (DeepMind) | 2019/2020 | 1D Conv 压缩 | [compressive-transformers.md](../references/compressive-transformers.md) | ✅ |
| 3 | **StreamingLLM** (MIT+Meta) | 2023/2024 | 滑动窗口 + attention sink | [streaming-llm.md](../references/streaming-llm.md) | ✅ |
| 4 | **InfLLM** (THU+MIT+Meta) | 2024 | 块级 memory + 训练无关 | [inf-llm.md](../references/inf-llm.md) | ✅ |
| 5 | **AutoCompressors** (Princeton) | 2023 | LLM 自压缩 + summary accumulation | [auto-compressors.md](../references/auto-compressors.md) | ✅ |
| 6 | **RMT** (MIPT+AIRI) | 2022 | 特殊 [mem] tokens + BPTT 跨段 | [rmt.md](../references/rmt.md) | ✅ |
| 7 | **Landmark Attention** (EPFL) | 2023 | attention 内生 block retrieval | [landmark-attention.md](../references/landmark-attention.md) | ✅ |

**完成度：Loop B 7/7 = 100%**（6 大流派 + 1 个 attention 内生检索代表）

### 7.2 已应用（11 个项目内部文档整合）

> 📌 **2026-07-31 更新**：从 4 项扩展至 11 项，覆盖 Loop B 调研 + PLT/Φ 集成 + 全项目 README 同步。

| # | 目标文件 | 整合内容 | Commit |
|---|---------|---------|--------|
| 1 | [hippo/retrieval-extraction.md](./hippo/retrieval-extraction.md) | 加入 InfLLM 块级检索参考 + 端侧回退路径 | `5b1ece0` |
| 2 | [logos/k-strategy.md](./logos/k-strategy.md) | 新增 §2.6 端侧 KV cache Loop B fallback | `24b6516` |
| 3 | [hippo/README.md](./hippo/README.md) | 新增 §2.6 与 Loop B 6 列对照表 | `9153ee2` |
| 4 | [sadko/open-issues.md](./sadko/open-issues.md) | 新增 B-09 Compressive 1D Conv 备选 | `03c0a26` |
| 5 | [references/README.md](../references/README.md) | 索引登记 7 篇 Loop B 论文 + §3.9 速查表 | `0d0ddd1` + `e527d46` |
| 6 | [logos/64m-validation-plan.md](./logos/64m-validation-plan.md) | §3.6 Loop B 备选实验（条件启动）| `d63e26c` |
| 7 | [logos/64m-validation-plan.md](./logos/64m-validation-plan.md) | §4.6 L-PLT 评估 + §4.6.9 L-PLT+Φ 联合实验 | `1cb53dd` + `77eaa8c` |
| 8 | [logos/k-strategy.md](./logos/k-strategy.md) | §2.2 PLT 机制澄清 + §2.2.7 PLT+Φ 集成设计 | `90016a0` + `b026847` |
| 9 | [logos/k-strategy.md](./logos/k-strategy.md) | §2.2.7 PLT + Φ（DiscoLoop 对齐 vs PLT 跨位置对齐）| `b026847` |
| 10 | [README.md](../../README.md) + AGENTS.md + architecture.md | 四研究线格局 + Loop B 谱系全项目同步 | `a531a3d` ~ `5437dd9` |
| 11 | [four-lines-synthesis.md](./four-lines-synthesis.md) + rfcs + papers | 四线综合脉络 + RFC/论文索引同步 | `ec31d8d` ~ `5c7e3a7` |

### 7.3 待办（按需追加）

| 候选 | 优先级 | 备注 |
|------|:---:|------|
| 关注 `[Munkhdalai 2022]` 等**Memorizing Transformers 系列后续** | 🟢 | 与本项目关联弱 |
| 关注 Mamba / SSM 类的**状态空间记忆**方案 | 🟢 | 与 Transformer 范式正交 |
| 关注"无限上下文"领域的 Sora / Gemini 1.5 等工业实践 | 🟢 | 闭源，无法独立验证 |

> Loop B 调研已**完整覆盖** 6 大流派（kNN 检索 / 1D Conv 压缩 / 滑动窗口+anchor / 块级检索 / LLM 自压缩 / 特殊 tokens / attention 内生）。后续若有新论文，**仅当项目出现新需求时**再调研。

---

**最后更新**：2026-07-31（v1.2：§1.5 新增 DiscoLoop × PLT 对齐统一视角 + §7 完整更新 Loop B 调研完成度 7/7 + 应用整合 4 项 + 待办 3 项）
**作者**：来自 DiscoLoop 调研 + 用户概念澄清工作流
**关键澄清**：用户理解的"一个 decode 输出 = 一个循环"在循环 Transformer 中是错误的；正确是 K 次循环迭代。详见 §0.3