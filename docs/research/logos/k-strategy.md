# Logos 端侧并行化与 K 值策略

> **一句话定位**：Logos 端侧部署的并行化策略——K 值是**超参数**（不是架构决策），端侧 K>4 串行无价值，需用 PLT / Radix Cache / 层次化等并行化策略突破 K 限制。
> **性质**：Logos 主线的核心策略文档
> **最后更新**：2026-07-31（v1.3：新增 §2.6 端侧 KV cache Loop B fallback，引用 streaming-llm / inf-llm）

---

## 0. 核心结论

| 决策 | 结论 |
|------|------|
| **循环架构方向** | ✅ LoopCoder-v2 验证正确（K=2 > K=1 +50%），方向已定 |
| **K 值定位** | **超参数**（不是架构决策），由 64M 验证决定 |
| **K 默认策略** | **不预设固定 K**，用 Per-Token 早退自适应 |
| **并行化策略** | **三件套**：PLT/HLT-PLT + Radix Cache + 层次化 |
| **GR AM 集成** | ❌ 不集成，仅作架构参考 |

---

## 1. K 值的合理定位

### 1.1 K 值是超参数，不是架构决策

> **关键澄清（2026-07-29 v1.2）**：之前的文档过度强调了具体 K 值（K=2 默认等）。**这是错误的——K 是超参数**。

K 值的重要性被高估：
- K=2 > K=1（LoopCoder-v2 实证 +50%）说明**循环有用**
- K=3+ 在 PLT 上崩溃（在 HRM 上不一定）说明**循环有上限**
- K 实际是 **"循环架构 + 任务 + 规模"** 的复合函数
- 我们从零训练，K 应该按需调整，而不是固定

### 1.2 K 的实际意义

```
K = 实际循环次数 / 共享块参数

含义：每次循环都是对前一状态的"反思-精炼"
约束：K 越大，延迟越长（端侧 K>4 串行 ≈ 0 价值）
自由：K 是设计旋钮，不是硬约束
```

**Logos 的策略**：**K 由训练时验证决定 + 推理时 Per-Token 早退自适应**。

### 1.3 为什么过度关注 K 是错的

| 错误观点 | 实际情况 |
|---------|---------|
| ❌ "K 必须等于 2 才是端侧可行的" | ✅ K 端侧可行 = 用并行化策略突破 |
| ❌ "K=8 是 HRM 的标准配置，Logos 必须用 K=8" | ✅ K 是超参，HRM 论文 K=8 是其选择，我们可不同 |
| ❌ "K-sweep 是前置必做的实验" | ✅ K-sweep 是 64M 验证的一部分，不是独立阻塞实验 |

---

## 2. 端侧并行化策略

### 2.1 为什么 K>4 串行循环无价值

| K | 单 token 延迟 | 应用场景 |
|:---:|:---:|---|
| K=1 | ~50ms | ✅ 自动驾驶感知 |
| K=2 | ~100ms | ✅ 实时对话 |
| K=4 | ~200ms | ⚠️ 仅离线规划 |
| K=8 | ~400ms | ❌ 完全不可用 |

**核心问题**：K>4 在端侧 = 用不起 = 实际价值≈0。**无论 K=2/4/8 哪个最优，串行 K>4 都不可用**。

**解决思路**：用并行化策略让 K=4-8 在端侧预算内可用。

### 2.2 PLT/HLT-PLT 并行循环

**借鉴自**：PLT（[loopcoder-v2.md §3](../references/loopcoder-v2.md#3-核心创新)）

**移植到 HRM（H/L 风格）**：HLT-PLT

```python
# HLT-PLT（H 串行 + L 并行）
def hlt_plt(x, H_cycles=2, L_cycles=3):
    h_H = prefill(x)
    h_L = h_L_init.expand_as(h_H)
    
    for h in range(H_cycles):
        # L 循环并行（CLP 偏移打破串行依赖）
        h_L = parallel_L_chain(h_L + h_H)
        # H block 串行
        h_H = H_block(h_H + h_L)
    
    return h_H
```

**延迟节省**：8 步变 3 步，**延迟降低 62%**。

**适用场景**：单轨迹推理 K=4-8 时。

### 2.3 Radix Cache 多路径并行

**核心思想**：多路径并行采样，共享前缀通过 Radix Tree 缓存。

```python
def radix_cache_multipath(x, n_paths=4, K=2):
    base_h = prefill(x)  # 共享前缀
    radix_cache = RadixCache(base_h)
    
    # N 条路径并行采样
    paths = []
    for i in range(n_paths):
        h_i = base_h.clone()
        for k in range(K):
            cached = radix_cache.get(h_i.hash())
            if cached is not None:
                h_i = cached
                continue
            h_i = hrm_block(h_i, x)
            radix_cache.put(h_i.hash(), h_i)
        paths.append(h_i)
    
    return aggregate(paths)
```

**延迟**：N=4 路径 × K=2 = 8 步推理，延迟 ≈ **1 次 K=8 forward**（路径并行）。

**适用场景**：多假设决策（路口左/右转）。

### 2.4 层次化推理

**核心思想**：主推理在关键节点暂停，派生子推理完成局部任务，再恢复。

```python
def hierarchical_reasoning(x, K_main=2, K_sub=1):
    h = prefill(x)
    
    # 主推理第一阶段
    h = hrm_main_block(h, x)
    
    # 触发子推理（并行）
    sub_results = parallel([
        sub_reasoning_1(h, K=K_sub),
        sub_reasoning_2(h, K=K_sub),
    ])
    
    # 主推理第二阶段（融合子结果）
    h = hrm_main_block(h, x, sub_context=sub_results)
    return h
```

**延迟**：K_main + max(K_sub) = 2 + 1 = 3 次 forward。

**适用场景**：可分解推理（数学先计算再验证）。

### 2.5 三方案对比

| 场景 | PLT/HLT-PLT | Radix Cache 多路径 | 层次化 |
|------|:---:|:---:|:---:|
| **单轨迹推理 K=4-8** | ✅ 端侧化 | — | — |
| **多假设决策** | — | ✅ 端侧化 | — |
| **可分解推理** | — | — | ✅ 主 + 子并行 |
| **延迟节省** | 62% | N 倍 | 2-3 倍 |

**结论**：**三方案互补，按场景选择，不是互斥**。

---

### 2.6 端侧 KV cache 备选方案（Loop B fallback）

> 📌 **2026-07-31 新增**：基于 Loop B 文献调研（[loop-memory-survey.md §2.2](../loop-memory-survey.md)），补充端侧 KV cache 管理的**最低成本 fallback**。当上述三方案都无法满足端侧预算时使用。

#### 2.6.1 StreamingLLM 滑动窗口（最轻量级 fallback）

**核心思想**（参考 [streaming-llm.md §3](../references/streaming-llm.md)）：保留 4 个初始 token 作为 attention sink + 最近 N tokens 的滑动窗口。

```python
# 端侧 Logos 的最低成本 KV 管理
def streamingllm_kv_cache(x, sink_size=4, window_size=2048):
    # x = 当前输入
    kv_sinks = cache[:sink_size]          # 4 个初始 token（永久保留）
    kv_window = cache[-window_size:]     # 最近 2048 tokens
    kv_combined = concat([kv_sinks, kv_window])
    return kv_combined
```

**关键优势**：
- **完全训练无关**（无需修改 Logos）
- **O(1) 显存**（不随序列长度增长）
- **22.2× 加速** vs sliding window with re-computation（论文实测）
- 已在 Llama-2-7B/13B/70B, MPT-7B/30B, Falcon-7B/40B, Pythia-2.8B/6.9B/12B 验证

**关键限制**：
- ❌ **不解决"真无限上下文"**——evicted token 不可恢复
- ❌ 不解决"远距离实体召回"（与 Memorizing Transformers 正交）
- ✅ 仅解决"流式部署"工程问题

**Logos 适用场景**：
- 1B 端侧部署（v3.0 之后）
- 当 Hippo FSQ 端侧成本过高时的 fallback
- 实时多轮对话（Loop C 维度）

#### 2.6.2 端侧三档方案对比

| 方案 | 显存 | 加速比 | 远距离能力 | 训练成本 | 端侧复杂度 | 推荐阶段 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| **StreamingLLM 滑动窗口** | O(1) | 22.2× | ❌ | 0 | 🟢 极低 | 1B+ 推理 fallback |
| **InfLLM 块级 memory** | O(M) | 中等 | ✅ | 0 | 🟡 中 | 1B+ 推理 |
| **Hippo FM + FSQ** | O(M) | 取决于 FM | ✅✅ | 高 | 🔴 高 | 训练（端到端可微）|
| **PLT/HLT-PLT**（§2.2）| O(K × L) | 62% | — | 中 | 🟡 中 | Logos 1B 训练+推理 |
| **Radix Cache**（§2.3）| O(N) | N 倍 | — | 中 | 🟡 中 | Logos 1B 多路径 |

**决策原则**：
- **训练时**：Hippo FM + FSQ（端到端可微）
- **推理时（端侧预算紧张）**：StreamingLLM → InfLLM → Hippo（按硬件能力选）
- **推理时（云端）**：Hippo + Radix Cache

---

## 3. Per-Token 早退（动态 K）

**借鉴自**：[per-token-convergence.md §3](../references/per-token-convergence.md#3-核心发现)（90% token 6 步收敛）

```python
def per_token_early_exit(x, max_K=8, epsilon=1e-4):
    h = init_state(x)
    exit_mask = torch.zeros(B, L, dtype=torch.bool)
    
    for k in range(max_K):
        h_new = hrm_block(h, x)
        delta = (h_new - h).norm(dim=-1)  # [B, L]
        
        new_exit_mask = delta < epsilon
        h = torch.where(new_exit_mask.unsqueeze(-1), h_new, h)
        exit_mask = exit_mask | new_exit_mask
        
        if exit_mask.all():
            break
    
    return h, exit_mask
```

**延迟节省**：平均循环数 ≤ 3（基于 Per-Token 数据）。

**Logos 设计哲学**：**K 应该是运行时决策，不是设计时固定值**。

---

## 4. 为什么不在 Logos 集成 GRAM

**关键澄清（2026-07-29）**：

| 维度 | GRAM 论文 | Logos 设计 |
|------|---------|----------|
| **多轨迹机制** | μ, σ 可学习 + ε ~ N(μ, σ²I) | Per-Token 早退 + Radix Cache |
| **变分训练** | ELBO loss | 不使用 ELBO |
| **训练流程** | 变分注入到 H 模块 | 直接回归目标 |
| **依赖** | 需要变分训练基础设施 | 不需要 |

**为什么 Logos 不集成 GRAM**：
- Logos 是全新架构，从零训练——**不需要 GRAM 的训练流程**
- GRAM 的"多轨迹综合决策"**思想**有用，但实现可以完全独立
- Logos 用 Per-Token 早退 + Radix Cache + 层次化实现"测试时 scaling"，更端侧友好
- 不被 GRAM 的变分训练流程绑定

**GR AM 仅作**：
- 架构灵感（多轨迹综合决策的思想）
- 参考论文（在 [references/gram.md](../references/gram.md) 注明）

---

## 5. 与 64M 验证的衔接

**64M 验证的关键产出**：
- 不追求"最佳 K 值"
- 追求"K 范围 + Per-Token 早退有效性 + 并行化策略可行性"
- 《64M 机制清单》报告 K 在不同规模、不同任务下的范围

**64M v1.0-v3.0 验证的 K 相关实验**：
- v1.0：双时间尺度对比（HRM H/L vs Split-GQA vs 混合）
- v2.0：循环策略对比（串行 vs PLT vs 早退 vs Radix vs 层次化）
- v3.0：多轨迹并行决策（不集成 GRAM，独立实现）

详细见 [64m-validation-plan.md](./64m-validation-plan.md)。

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| **PLT 位置偏移税** | 🟠 中 | 高 | STARS 谱正则化（[stars.md](../references/stars.md)）|
| **Radix Cache 实现复杂** | 🟠 中 | 中 | 先用 PyTorch 原生，v1.0 再优化 |
| **层次化难以端到端训练** | 🟡 低 | 中 | 退回单层推理 + 多路径 |
| **Per-Token 早退阈值敏感** | 🟡 低 | 低 | 简单网格搜索即可 |

---

## 7. 相关文档

| 文档 | 关系 |
|------|------|
| [whitepaper.md](./whitepaper.md) | Logos 主线白皮书（本文档的父级）|
| [roadmap.md](./roadmap.md) | 详细路线图 |
| [64m-validation-plan.md](./64m-validation-plan.md) | 64M 验证计划 |
| [v2-architecture.md](./v2-architecture.md) | v2.0 多种循环策略详细 |
| [v3-architecture.md](./v3-architecture.md) | v3.0 多轨迹并行详细 |
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | PLT 架构 |
| [docs/references/per-token-convergence.md](../references/per-token-convergence.md) | 早退证据 |
| [docs/references/stars.md](../references/stars.md) | 崩溃修复 |
| [docs/references/streaming-llm.md](../references/streaming-llm.md) | 端侧 KV fallback（§2.6）|
| [docs/references/inf-llm.md](../references/inf-llm.md) | 块级 memory 备选（§2.6）|
| [../research/loop-memory-survey.md](../research/loop-memory-survey.md) | 循环+记忆系统完整谱系 |

---

**最后更新**：2026-07-29（v1.2 重大调整）
**作者**：来自工作流（Logos K 策略整理）
**版本**：v1.2