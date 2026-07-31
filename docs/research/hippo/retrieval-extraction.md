# Hippo Flow Matching 检索提取（FM Retrieval & Extraction）

> **方向**：方向 3 / 3（记忆 / KG / 检索）
> **状态**：🆕 骨架（待填充具体机制设计）
> **上游 spec**：[hippo/README.md §1](./README.md#1-三个并行研究方向)

---

## 0. 核心问题

FM 的 ODE 可逆性如何用于检索？ODE 方向如何选择？Top-K 精度与速度权衡？碎片化关联如何提取？

---

## 1. 研究目标

基于 Flow Matching 的检索模块：
- Recall@K ≥ 0.9
- 端侧 < 5ms
- 支持模糊跨域联想

---

## 2. 最小验证单元

`FMRetrieval` 模块：
- (a) 输入 query 向量
- (b) 输出 Top-K 码字 + 关联 KV
- (c) 支持 Slerp 模糊联想

---

## 3. 关键指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| Recall@K | Top-K 命中率 | ≥ 0.9 |
| 检索延迟 | 端到端 query → result | < 5ms |
| 联想质量 | Slerp 跨码字解码合理性 | 人工评估 + 自动 |
| 失败降级率 | 检索失败时静默退化率 | 100% |

---

## 4. 前置依赖

- [`sadko/elf-phase0-manual.md`](../sadko/elf-phase0-manual.md)（FM+FSQ 基础）
- 可选：[`memory-architecture.md`](./memory-architecture.md) 的码字 schema
- 可选：[`graph-growth.md`](./graph-growth.md) 的图查询 API

---

## 5. 待填充内容（实施时）

- [ ] FM 检索算法（ODE 方向选择策略：query → 码字 vs 码字 → query）
- [ ] Slerp 联想测试（跨码字解码为连续向量）
- [ ] 失败降级协议（空 KV + 全 0 嵌入的静默降级路径）
- [ ] 参考实现（PyTorch 模块 + 单元测试）
- [ ] 验证实验（最小验证 / 消融 / 超参扫描 / 多数据集 / 接口契约）

---

## 6. 鲁棒性评分（待填）

[待实施后按 hippo/README.md §3.3 评分]

---

**下一步**：调用 writing-plans skill 为本方向生成详细实施计划。
