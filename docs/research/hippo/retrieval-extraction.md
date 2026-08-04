# Hippo Flow Matching 检索提取（FM Retrieval & Extraction）

> **方向**：方向 3 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)
> **外部参考**：[InfLLM 块级检索](../../references/inf-llm.md)（Loop B 块级 memory 范式）+ [Memorizing Transformers](../../references/memorizing-transformers.md)（kNN 检索范式）+ [Compressive Transformers](../../references/compressive-transformers.md)（压缩范式）

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
- 外部参考：[InfLLM §6](../../references/inf-llm.md)、[Memorizing Transformers §6](../../references/memorizing-transformers.md)、[Compressive Transformers §6](../../references/compressive-transformers.md)（与本项目架构对照）

---

## 5. 待填充内容（实施时）

- [ ] FM 检索算法（ODE 方向选择策略：query → 码字 vs 码字 → query）
- [ ] Slerp 联想测试（跨码字解码为连续向量）
- [ ] 失败降级协议（空 KV + 全 0 嵌入的静默降级路径）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）
- [ ] **块级检索备选**（参考 [InfLLM §3.1](../../references/inf-llm.md)）：若 64M 阶段 FSQ 码本利用率 < 80%，降级评估
- [ ] **端侧回退路径**：FM → 1D Conv（[Compressive §3.2](../../references/compressive-transformers.md)）→ 滑动窗口（[StreamingLLM §3.2](../../references/streaming-llm.md)）三级回退

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
