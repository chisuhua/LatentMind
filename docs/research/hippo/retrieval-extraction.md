# Hippo Flow Matching 检索提取（FM Retrieval & Extraction）

> **方向**：方向 3 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)
> **外部参考**：[InfLLM 块级检索](../../references/inf-llm.md)（Loop B 块级 memory 范式）+ [Memorizing Transformers](../../references/memorizing-transformers.md)（kNN 检索范式）+ [Compressive Transformers](../../references/compressive-transformers.md)（压缩范式）+ [Landmark Attention](../../references/landmark-attention.md)（attention 内生检索范式）+ [RMT](../../references/rmt.md)（特殊 memory tokens 接口范式）

---

## 0. 核心问题

FM 的 ODE 可逆性如何用于检索？ODE 方向如何选择？Top-K 精度与速度权衡？碎片化关联如何提取？

**额外问题（来自 Loop B 文献调研）**：
- 检索粒度：token-level vs block-level vs code-level？
- 训练时压缩 vs 推理时检索：FM 压缩是否需要在每段重做？
- 与 Loop B 范式（检索/压缩/滑动窗口）的对比取舍？

---

## 1. 研究目标

基于 Flow Matching 的检索模块：
- Recall@K ≥ 0.9
- 端侧 < 5ms
- 支持模糊跨域联想

**端侧备份方案（Plan B）**：
- 若 FM 压缩在 64M 阶段端侧成本过高，回退到 **1D Conv 压缩**（参考 [Compressive Transformers §3.2](../../references/compressive-transformers.md)）
- 若端侧 KV cache 检索延迟超预算，回退到 **StreamingLLM 滑动窗口 + attention sink**（参考 [streaming-llm.md](../../references/streaming-llm.md)）

---

## 2. 最小验证单元

`FMRetrieval` 模块：
- (a) 输入 query 向量
- (b) 输出 Top-K 码字 + 关联 KV
- (c) 支持 Slerp 模糊联想

**候选检索粒度对比**（来自 Loop B 文献）：
- **Token-level（kNN）**：参考 [Memorizing Transformers](../../references/memorizing-transformers.md) §3.1 — 端侧不可行
- **Block-level**：参考 [InfLLM](../../references/inf-llm.md) §3.1 — 折中方案
- **Code-level（FSQ 码字）**：**Hippo 当前方案** — 信息密度最高
- **无检索（滑动窗口）**：参考 [StreamingLLM](../../references/streaming-llm.md) §3.2 — 极简 fallback
- **Attention 内生检索**：参考 [Landmark Attention](../../references/landmark-attention.md) §3.1 — **候选 Plan B**（详见 §2.1）

### 2.1 候选 Router 设计（基于 Landmark Attention 思想）

> 📌 **2026-07-31 新增**：参考 [Landmark Attention §3.1](../../references/landmark-attention.md)，Hippo Router 的 4 种候选实现：

| 方案 | 描述 | 端侧 | 推荐度 |
|------|------|:---:|:---:|
| **A. 显式 kNN** | 与 FSQ 码字计算 cosine 距离，选 top-K | ❌ kNN 检索重 | 低 |
| **B. 显式 Top-K**（当前）| 与所有 FSQ 码字计算 similarity | ⚠️ O(K × d) | 中 |
| **C. Attention 内生检索**（Landmark 风格）| 训练 attention 自身学会"用 landmark 评估 block 相关性" | ✅ 仅注意力 | **高** |
| **D. 训练无关 kNN**（Memorizing 风格）| kNN 不可微，端侧不可行 | ❌ | 低 |

**方案 C 的具体实现**（参考 Landmark Attention §3）：

```python
# 方案 C: Attention 内生检索（Landmark 风格）
def landmark_router(query, fsq_codes, k_top=8):
    """
    query: 当前 token 的 hidden state
    fsq_codes: 256 个 FSQ 码字的 key vector
    k_top: 检索的 top-k 码字数

    # 1. 计算 query 与所有 FSQ 码字的 attention score
    scores = einsum("d,cd->c", query, fsq_code_keys) / sqrt(d)

    # 2. Top-k 选择（与 Landmark 的 top-k block retrieval 同构）
    top_scores, top_indices = scores.topk(k_top)

    # 3. 用 top-k 码字的 value（KV 嵌入）做加权
    weights = softmax(top_scores)
    retrieved_kv = einsum("k,kd->d", weights, fsq_code_values[top_indices])

    return retrieved_kv, top_indices, top_scores  # 含置信度
```

**与现有方案 B 的差异**：
- B 显式计算所有 256 个码字 → 端侧 O(256 × d) 计算
- C **端到端可微 + 可解释**——可看到 attention 选中了哪些码字
- C 训练时让 attention 学会"用码字代表语义"——避免显式 kNN 距离计算

**关键启示**（从 Landmark LLaMA 7B 32K 实验）：
- ✅ **fine-tune 即可让 attention 学会新检索模式**——无需从头训练
- ✅ **可解释性强**——attention score 直接显示选了哪些码字
- ⚠️ 训练时需"land-mark"对应——FSQ 码字已是离散锚点，天然适合作为 landmark
- ❌ **不**完全替换显式 Router——RetAIN 显式 K 选以满足 I3 "含置信度"契约

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| Recall@K | Top-K 命中率 | ≥ 0.9 |
| 检索延迟 | 端到端 query → result | < 5ms |
| 联想质量 | Slerp 跨码字解码合理性 | 人工评估 + 自动 |
| 失败降级率 | 检索失败时静默退化率 | 100% |
| **码本使用率**（新增）| 实际使用码字 / 总码字 | ≥ 80%（参考 [Hippo README §3.3 鲁棒性评分](./README.md#33-鲁棒性评分体系-0-10)）|
| **块级检索命中率**（备选）| InfLLM 风格的 block-level top-k 命中率 | ≥ 0.7（端侧备选验证）|

---

## 4. 前置依赖

- [`sadko/hippo-phase0-manual.md`](../sadko/hippo-phase0-manual.md)（FM+FSQ 基础）
- 可选：[`memory-architecture.md`](./memory-architecture.md) 的码字 schema
- 可选：[`graph-growth.md`](./graph-growth.md) 的图查询 API
- 外部参考：[InfLLM §6](../../references/inf-llm.md)、[Memorizing Transformers §6](../../references/memorizing-transformers.md)、[Compressive Transformers §6](../../references/compressive-transformers.md)、[Landmark Attention §6](../../references/landmark-attention.md)、[RMT §6](../../references/rmt.md)（与本项目架构对照）

---

## 5. 待填充内容（实施时）

- [ ] FM 检索算法（ODE 方向选择策略：query → 码字 vs 码字 → query）
- [ ] Slerp 联想测试（跨码字解码为连续向量）
- [ ] 失败降级协议（空 KV + 全 0 嵌入的静默降级路径）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）
- [ ] **块级检索备选**（参考 [InfLLM §3.1](../../references/inf-llm.md)）：若 64M 阶段 FSQ 码本利用率 < 80%，降级评估
- [ ] **端侧回退路径**：FM → 1D Conv（[Compressive §3.2](../../references/compressive-transformers.md)）→ 滑动窗口（[StreamingLLM §3.2](../../references/streaming-llm.md)）三级回退
- [ ] **Landmark-style Router 评估**（参考 [Landmark §3.1](../../references/landmark-attention.md) §2.1 方案 C）：与当前显式 Top-K 对比端侧延迟与可解释性
- [ ] **RMT 特殊 token 接口**（参考 [RMT §3.1](../../references/rmt.md)）：为 Hippo 增设 `<MemRead>` `<MemWrite>` 特殊 token 候选

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
