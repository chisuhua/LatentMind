# StreamingLLM 参考资料

> **论文**：*Efficient Streaming Language Models with Attention Sinks*
> **作者**：Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
> **机构**：MIT-HAN-Lab + Meta AI + CMU
> **arXiv**：2309.17453v2（2023-09-29）
> **会议**：ICLR 2024
> **代码**：[github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)（6.2k+ stars）
> **性质**：**Loop B（inter-decode 记忆压缩）的最简化代表**——发现"attention sink"现象，证明**4 个初始 token + 滑动窗口**即可达到 4M tokens 稳定推理

---

## 1. 一句话定位

**发现"attention sink"现象——LLM 总是把不成比例的注意力分配给初始 token（即使它们无语义重要性）。基于此，保留 4 个初始 token 的 KV + 最近 N 个 token 的滑动窗口，可让 Llama-2/Falcon/MPT/Pythia 在 4M tokens 上稳定推理，**比 sliding window with re-computation 快 22.2×**。**完全训练无关**。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），是**最轻量级**的记忆方案

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2309.17453 |
| GitHub | https://github.com/mit-han-lab/streaming-llm |
| 论文网站 | https://hanlab.mit.edu/projects/streamingllm |
| ICLR 2024 Slides | https://iclr.cc/media/iclr-2024/Slides/18794.pdf |

---

## 3. 核心创新

### 3.1 Attention Sink 现象（核心发现）

> "Beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads."

**精确解释**：
- Softmax 要求 attention scores 总和为 1
- 当 query 与许多历史 token 都不强相关时，模型**必须**把 attention 分配到**某处**
- 初始 token 因"对所有后续 token 可见"（自回归特性）成为天然的"垃圾收集处"
- 即便初始 token 无语义重要性，它们仍承担"吸收不需要的 attention"的角色

### 3.2 StreamingLLM 架构

**KV cache 分两部分**：

```
┌─────────────────────────────────────┐
│  KV Cache (固定大小)                │
│  ┌──────────┬──────────────────────┐│
│  │ Sinks    │ Rolling Window       ││
│  │ (4 tokens)│ (most recent N)      ││
│  └──────────┴──────────────────────┘│
└─────────────────────────────────────┘
     ↑ always kept       ↑ FIFO
     ↑ 无需重计算        ↑ 无需重计算
```

**关键特性**：
- **完全训练无关**（training-free）
- **位置编码用相对位置**（如 cache 内位置 0-7），不用原始位置
- 支持 RoPE / ALiBi 等相对位置编码
- 与现有推理框架（HuggingFace）兼容

### 3.3 关键实验

| 任务 | StreamingLLM | 滑动窗口 + 重计算 | dense attention |
|------|:---:|:---:|:---:|
| PG-19 4M tokens 推理 | **稳定** | 慢 22.2× | OOM |
| 模型族 | Llama-2-[7,13,70]B / MPT-[7,30]B / Falcon-[7,40]B / Pythia-[2.8,6.9,12]B | 同 | 同 |
| 训练开销 | 0 | 0 | 重计算 N× |
| 显存 | O(N) | O(N) | O(序列长度) |
| PPL 与 sliding + re-comp | 持平 | 基线 | 不适用 |

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| Attention sinks 数量 | **4 个初始 token** | 1-2 个不够 |
| Rolling window 大小 | 1024-2048（典型）| 与训练窗口匹配 |
| 最大稳定序列 | **4M tokens** | 4 模型族验证 |
| 加速比 | **22.2×** | vs sliding window with re-computation |
| 显存 | O(N)（固定）| 与 sliding window 持平 |
| iPhone 部署 | ✅ | 论文有演示 |
| **对 INF 上下文支持** | ❌ | 上下文窗口不变（仍受训练限制）|

---

## 5. 关键发现

### 5.1 Attention Sink 的位置 vs 语义

> "Adding initial four '\n's can also recover perplexity. Therefore, it is position!"

**意义**：attention sink 的角色**与位置绑定**，与语义无关。可以使用任意 4 个初始 token（即使是空行）作为 sink。

### 5.2 不支持"真正无限上下文"

> "Can StreamingLLM give us infinite context? Non-stop chatting ≠ Infinite context. Tokens that are evicted from cache cannot be attended."

**意义**：StreamingLLM 解决**工程问题**（流式部署），**不**解决**长程语义**问题。evicted 的 token 不可恢复地丢失。

### 5.3 训练 vs 推理的不对称

> "Vanilla model requires the addition of multiple tokens as attention sinks to maintain stable streaming perplexity. In contrast, the model trained with a sink token achieves satisfactory streaming performance using just the sink token."

**意义**：预训练时显式添加 sink token，推理时只需 1 个 sink。这是**可选的训练技巧**。

---

## 6. 与本项目架构的对照

| 维度 | StreamingLLM | 本项目 Hippo | 关键差异 |
|------|--------------|-------------|----------|
| **存储内容** | 4 初始 token + 滑动 N token | FSQ 256 码本 | Hippo 信息密度高 |
| **压缩** | ❌ 完全不压缩 | ✅ FM 压缩 | Hippo 显式压缩 |
| **检索** | 无（仅 attention）| Top-K 码字 | Hippo 主动检索 |
| **可微** | 不适用（训练无关）| ✅ | StreamingLLM 不需训练 |
| **端侧友好** | ✅✅ | ⚠️ | StreamingLLM 极轻量 |
| **目标** | 流式部署 | 知识召回 + 多模态 | Hippo 范围更广 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ **"4 个 sink token" 思想可借鉴到 Logos 系统提示**——保留 4 个 KV 作为跨 token 的"注意力汇聚点"
- ✅ **训练无关 + 22.2× 加速**证明 Loop B 记忆系统的**端侧可行性**
- ✅ **滑动窗口 + attention sink** 可作为端侧 Logos 的**最低成本记忆方案**
- ⚠️ StreamingLLM 不解决"远距离实体召回"——这仍是 Hippo 的事

---

## 7. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| 不能恢复 evicted token | 🔴 高 | 流式部署 ≠ 真无限上下文 |
| 仅 autoregressive LLM | 🟠 中 | encoder-decoder 未测 |
| 4 个 sink 是经验值 | 🟡 低 | 论文未深入解释为什么 4 个最优 |
| 上下文窗口不变 | 🟠 中 | 仍受训练窗口限制 |
| 4M tokens 稳定但未在更长测试 | 🟡 低 | 论文未明确上限 |
| 不处理"远距离实体" | 🟠 中 | 与 Memorizing Transformers 正交 |

---

## 8. 关键引用块

> **核心架构**：
> "StreamingLLM exploits the fact that attention sinks have high attention values, and preserving them can maintain the attention score distribution close to normal. Therefore, StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV to anchor the attention computation and stabilize the model's performance."

> **加速比**：
> "StreamingLLM achieves up to 22.2× speedup, realizing the streaming use of LLMs."

> **未来方向**：
> "StreamingLLM firstly decouples the LLM's pre-training window size and its actual text generation length, paving the way for the streaming deployment of LLMs."

---

## 9. 相关工作

| 名称 | 与 StreamingLLM 关系 |
|------|---------------------|
| [Memorizing Transformers](./memorizing-transformers.md) | 互补：Memorizing 解决远距离实体，StreamingLLM 解决流式部署 |
| [Compressive Transformers](./compressive-transformers.md) | 早期路线：压缩而非滑动窗口 |
| [InfLLM](./inf-llm.md) | 后作：滑动窗口 + 块级 memory，更通用 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）|
| [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的对比参考 |

---

## 10. 对 LatentMind 的启示

### 10.1 短期（64M 阶段）

- ✅ **绝对不引入 kNN 或压缩机制**——64M 强约束
- ✅ 关注 Logos 64M 验证中的 **KV cache 显存预算**——StreamingLLM 的"固定 KV"思想是端侧基线

### 10.2 中期（300M 阶段）

- 🔍 评估 **StreamingLLM 风格的滑动窗口**作为 Logos KV cache 管理的**最简方案**
- 🔍 若 Hippo FSQ 验证未达预期，**StreamingLLM + Logos H/L** 是低风险 fallback

### 10.3 长期（1B+ 阶段）

- 🎯 **端侧部署核心方案**：StreamingLLM + 循环 Transformer + Hippo 三件套
- 🎯 **系统提示锚定**：4 个 sink token 可作为 Logos 系统提示的"注意力汇聚点"

---

**最后更新**：2026-07-31
**信息源**：arXiv 2309.17453v2 + ICLR 2024 Slides + GitHub 仓库
**作者**：来自 Loop B 文献谱系调研