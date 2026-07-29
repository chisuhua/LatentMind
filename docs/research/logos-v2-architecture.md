# Logos-Native-64M v2.0 架构设计：多种循环策略对比

> **一句话定位**：在 v1.0 最优基座（A.3 混合风格应最优）上对比五种循环策略——串行、PLT、Per-Token 早退、Radix Cache 多路径、层次化推理，绘制"延迟-精度" Pareto 曲线。
> **上游文档**：[logos-64m-validation-plan.md §4](./logos-64m-validation-plan.md#4-v20多种循环策略对比)
> **下游文档**：[logos-v3-architecture.md](./logos-v3-architecture.md)（基于 v2.0 推荐循环策略）
> **最后更新**：2026-07-29（v1.2：K 值调整为超参数，去 GR AM 集成）

---

## 0. 核心命题

**K 值是超参数，端侧 K>4 串行无价值。哪种循环策略能突破 K 限制？**

> 📌 **关键澄清（2026-07-29）**：
> - K 是超参数，不是固定值（由 64M 验证决定）
> - 不集成 GRAM，仅借鉴多轨迹思想
> - 五种循环策略并存，按场景选择

五种候选策略：

| 策略 | 延迟 | 突破 K 限制的机制 | 适用场景 |
|------|------|---------------|---------|
| **B.1 串行 K** | K × 1x | 无（基线）| K ≤ 4 |
| **B.2 PLT/HLT-PLT** | ≈ 1x | 并行多步 | 单轨迹 K=4-8 |
| **B.3 Per-Token 早退** | 平均 ≤ 3 | 动态 K | 通用 |
| **B.4 Radix Cache 多路径** | ≈ 1x | 多路径并行 | 多假设决策 |
| **B.5 层次化推理** | K_main + max(K_sub) | 主 + 子并行 | 可分解推理 |

**核心论证**：详见 [logos-k-strategy.md §4](./logos-k-strategy.md#4-多种端侧并行化策略latency-killer)。

---

## 1. 设计目标

**找到 Logos 64M v2.0 的最优循环策略组合**——B.2 + B.3 + B.4 三件套应能覆盖所有场景。

**验收标准**：
- B.2 PLT K=8 并行延迟 < 1.5x 串行 K=2
- B.3 早退平均 K ≤ 3
- B.4 Radix Cache N 路径延迟 < 1.5x K=2
- B.5 层次化延迟 < 3x K=1
- **核心结论**：三件套组合覆盖所有场景

---

## 2. 基座假设

本计划假设 v1.0 出 A.3 混合风格为最优基座。详细配置见 [logos-v1-architecture.md §3.3](./logos-v1-architecture.md#33-a3-混合风格应最优)。

若 v1.0 出其他结论，基座相应调整。

---

## 3. 五种循环策略详细

### 3.1 B.1 串行 K 循环（基线）

**实现**：

```python
def serial_loop(x, K):
    h = prefill(x)
    for k in range(K):
        h = hrm_block(h, x)  # 同一 block 反复调用
    return h
```

**延迟**：K × 1x
**适用**：K ≤ 4 端侧场景

**注**：K 是超参数，由 64M 验证决定。

### 3.2 B.2 PLT/HLT-PLT 并行循环

**借鉴自**：PLT（[loopcoder-v2.md §3](../references/loopcoder-v2.md#3-核心创新)）

**移植到 HRM**：HLT-PLT（H 串行包 L 并行）

```python
# HLT-PLT（H 串行 + L 并行）
def hlt_plt(x, H_cycles=2, L_cycles=3):
    h_H = prefill(x)
    h_L = h_L_init.expand_as(h_H)
    
    for h in range(H_cycles):
        # L 循环并行（CLP 偏移打破串行依赖）
        for l in range(L_cycles):
            h_L_shifted = shift_right(h_L)
            h_L_input = h_L_shifted + h_H
            h_L = parallel_L_block(h_L_input)
        h_H = H_block(h_H + h_L)
    
    return h_H
```

**关键约束**：
- H 循环必须串行（H 是慢速全局）
- L 循环可并行（L 是快速局部）
- 需要 L_block 支持并行 chain

**延迟节省**：

| 配置 | 总循环步 | 串行延迟 | HLT-PLT 延迟 | 节省 |
|------|:---:|:---:|:---:|:---:|
| 2H×3L | 8 | 8x | **3x**（2 H + 1 L forward）| **62%** |
| 2H×4L | 10 | 10x | **3x** | 70% |

**风险**：CLP 位置偏移税可能导致 L 循环精度损失。

### 3.3 B.3 Per-Token Early Exit

**借鉴自**：[per-token-convergence.md §3](../references/per-token-convergence.md#3-核心发现)（90% token 6 步收敛）

**实现**：

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

**优势**：
- 通用（适用于所有任务）
- 简单（per-token 早退决策）
- 与 B.2 PLT 可叠加（PLT 内 early exit）

### 3.4 B.4 Radix Cache 多路径并行

**借鉴自**：你提出的方案（详见 [logos-k-strategy.md §2.3](./logos-k-strategy.md#23-radix-cache-多路径并行)）

**核心思想**：借鉴 PLT 思想，多条路径在 latent tree search 中并行采样，共享前缀通过 Radix Tree 缓存。

**实现**：

```python
class RadixCache:
    """简化版 Radix Tree 缓存"""
    def __init__(self, base_h):
        self.tree = RadixTreeNode(base_h)
    
    def get(self, prefix_hash):
        return self.tree.get(prefix_hash)

def radix_cache_multipath(x, n_paths=4, K=2):
    # Prefill 共享前缀
    base_h = prefill(x)
    radix_cache = RadixCache(base_h)
    
    # N 条路径并行采样
    paths = []
    for i in range(n_paths):
        h_i = base_h.clone()
        for k in range(K):
            # 检索 Radix Cache
            cached = radix_cache.get(h_i.hash())
            if cached is not None:
                h_i = cached
                continue
            h_i = hrm_block(h_i, x)
            radix_cache.put(h_i.hash(), h_i)
        paths.append(h_i)
    
    # 综合 N 条路径
    return aggregate(paths)  # 平均 / 投票 / 加权
```

**延迟**：N=4 路径 × K=2 = 8 步推理，延迟 ≈ **1 次 K=8 forward**（路径并行 + Radix 缓存）。

**适用场景**：多假设决策（路口左/右转、规划路径 A/B/C）。

### 3.5 B.5 层次化推理

**借鉴自**：你提出的方案（详见 [logos-k-strategy.md §2.4](./logos-k-strategy.md#24-层次化推理)）

**核心思想**：主推理在关键节点暂停，派生子推理完成局部任务，再恢复。

**实现**：

```python
def hierarchical_reasoning(x, K_main=2, K_sub=1):
    h = prefill(x)
    
    # 主推理第一阶段
    h = hrm_main_block(h, x)
    
    # 触发子推理（并行）
    sub_results = parallel([
        hrm_sub_block_1(h, x),
        hrm_sub_block_2(h, x),
        hrm_sub_block_3(h, x),
    ])
    
    # 主推理第二阶段（融合子结果）
    h = hrm_main_block(h, x, sub_context=sub_results)
    
    return h
```

**延迟**：K_main + max(K_sub) = 2 + 1 = 3 次 forward。

**适用场景**：可分解推理（数学先计算再验证）。

---

## 4. 五策略延迟-精度 Pareto 对比

### 4.1 实验设计

| 策略 | K | N 路径 | 总循环步 | 单 token 延迟 | GSM8K 精度 |
|------|:---:|:---:|:---:|:---:|:---:|
| B.1 串行 K=1 | 1 | 1 | 1 | 50ms | baseline |
| B.1 串行 K=2 | 2 | 1 | 2 | 100ms | +X% |
| B.1 串行 K=4 | 4 | 1 | 4 | 200ms | +Y% |
| B.1 串行 K=8 | 8 | 1 | 8 | 400ms | +Z% |
| B.2 PLT/HLT-PLT K=8 | 8（并行）| 1 | 8 | **~120ms** | +Z% |
| B.3 Per-Token 早退 K_max=8 | 平均 3 | 1 | ~3 | ~150ms | ~+Y% |
| B.4 Radix Cache 4 路径 K=2 | 2 | 4 | 8 | **~120ms** | 多假设决策 +M% |
| B.5 层次化 K_main=2, K_sub=1 | 3 | 1 | 3 | ~150ms | 可分解 +D% |

### 4.2 决策矩阵

```
延迟 < 100ms 预算：
  ✅ B.1 K=1（50ms）
  ✅ B.1 K=2（100ms）
  ✅ B.4 Radix Cache 4 路径 K=2（~120ms 接近预算）

延迟 < 200ms 预算：
  ✅ B.2 PLT/HLT-PLT K=8（~120ms）
  ✅ B.3 早退平均 K=3（~150ms）
  ✅ B.5 层次化（~150ms）

精度优先（离线）：
  ✅ B.1 K=4 或 K=8（200-400ms）
```

### 4.3 推荐组合

**核心三件套**：
- **B.3 Per-Token 早退**（通用加速，通用）
- **B.4 Radix Cache**（多假设决策）
- **B.2 PLT**（v1.0 升级时用于 K=4-8 推理）

**扩展**：
- **B.5 层次化**：特定可分解任务（数学/代码）
- **B.1 串行**：所有场景的基础参考

---

## 5. 实验设计

### 5.1 训练配置

```python
# 共享训练配置（基于 v1.0 A.3 基座）
training_config = {
    'base': 'logos_v1_a3_mixed',
    'lr': 1e-4,
    'warmup': 2000,
    'batch_size': 64,
    'seq_length': 2048,
    'total_tokens': 4_000_000_000,
}
```

### 5.2 评估指标

| 指标 | 目的 |
|------|------|
| **单 token 延迟**（ms）| 端侧预算 |
| **GSM8K 精度** | 推理质量 |
| **早退平均 K** | 早退有效性 |
| **Radix Cache 命中率** | 缓存效率 |
| **PPL（C4 验证集）** | 语言建模 |

### 5.3 必做消融

按优先级：

1. **基线对比**：B.1 K=1/2/4/8 on GSM8K
2. **PLT 验证**：B.2 PLT K=8 on GSM8K（验证 HLT-PLT 可行性）
3. **早退扫描**：B.3 ε ∈ {1e-3, 1e-4, 1e-5} 网格
4. **Radix Cache 扫描**：B.4 N ∈ {2, 4, 8} 路径数
5. **层次化扫描**：B.5 K_sub ∈ {1, 2, 3} 子推理深度
6. **三件套组合**：B.1 K=2 + B.3 早退 + B.4 Radix Cache 端到端测试
7. **延迟-精度 Pareto**：五策略在同一图上对比

---

## 6. 验收标准

### 6.1 单策略验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| B.1 K=2 精度 vs K=1 | ≥ +5% | ≥ +10% |
| B.1 K=4 vs K=2 | ≥ +2% | ≥ +5% |
| B.2 PLT K=8 并行延迟 | < 1.5x K=2 | < 1.2x K=2 |
| B.2 PLT K=8 精度 vs K=8 串行 | ≥ 95% | ≥ 98% |
| B.3 早退平均 K | ≤ 3 | ≤ 2.5 |
| B.3 早退精度 vs K=2 固定 | ≥ K=2 - 1% | ≥ K=2 |
| B.4 Radix Cache 4 路径延迟 | < 1.5x K=2 | < 1.2x K=2 |
| B.4 Radix Cache 4 路径决策质量 | ≥ K=2 + 5% | ≥ K=2 + 10% |
| B.5 层次化延迟 | < 3x K=1 | < 2.5x K=1 |
| B.5 层次化精度 vs K=4 | ≥ K=4 - 2% | ≥ K=4 |

### 6.2 组合验收

| 验收项 | 合格线 |
|-------|:---:|
| 三件套端到端延迟 | < 150ms |
| 三件套覆盖场景 | 简单推理 + 决策 + 多假设 |

---

## 7. 决策传递

### 7.1 v2.0 出结果后

| v2.0 结论 | v3.0 基线 |
|----------|---------|
| 三件套最优（B.3 + B.4 + B.2）| 用三件套作为 v3.0 基线 |
| B.2 PLT 显著有效 | v3.0 在三件套基础上加 PLT |
| B.5 层次化显著有效 | v3.0 在三件套基础上加层次化 |
| 单一方案占优 | 退化到该方案 |

### 7.2 失败回退

| v2.0 失败 | 回退 |
|----------|------|
| B.3 早退不 work | 用固定 K=2 + 离线 K=4 |
| B.4 Radix Cache 实现复杂 | v3.0 用单路径并行 |
| B.5 层次化难以端到端 | 不集成层次化 |

---

## 8. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-64m-validation-plan.md §4](./logos-64m-validation-plan.md#4-v20多种循环策略对比) | 本文档的父级 |
| [logos-v1-architecture.md](./logos-v1-architecture.md) | v1.0 双时间尺度对比（v2.0 基座）|
| [logos-k-strategy.md §2](./logos-k-strategy.md) | 五策略核心论证 |
| [logos-whitepaper.md §2](./logos-whitepaper.md) | Logos 白皮书 |
| [docs/rfcs/RDD-0001-k-sweep-experiment.md](../rfcs/RDD-0001-k-sweep-experiment.md) | K-sweep RFC（已合并到本文档）|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | PLT 架构详细（v2.0 B.2 借鉴）|
| [docs/references/per-token-convergence.md](../references/per-token-convergence.md) | 早退证据（v2.0 B.3 依据）|

---

**最后更新**：2026-07-29（v1.2 重大调整）
**作者**：来自工作流（Logos v2.0 架构设计）
**版本**：v1.2