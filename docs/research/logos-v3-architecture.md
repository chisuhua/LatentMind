# Logos-Native-64M v3.0 架构设计：多轨迹并行决策

> **一句话定位**：在 v2.0 推荐循环策略上集成多轨迹并行决策——借助 Radix Cache 多路径并行，让 N 条路径在端侧预算内完成（**不集成 GRAM**，独立实现）。
> **上游文档**：[logos-64m-validation-plan.md §5](./logos-64m-validation-plan.md#5-v30多轨迹并行决策)
> **下游文档**：交付《Logos 64M 验证报告》→ 启动 Logos 300M 训练
> **最后更新**：2026-07-29（v1.2：去除 GR AM 集成，明确独立实现多轨迹）

---

## 0. 关键定位：不集成 GRAM

> 📌 **2026-07-29 v1.2 重要澄清**：
> - **不集成 GRAM 训练流程**（μ, σ 可学习 + ELBO loss）
> - **独立实现多轨迹思想**：Per-Token 早退 + Radix Cache 多路径 + 层次化推理
> - GRAM 仅作**架构灵感参考**，借鉴"多轨迹综合决策"的思想
> - Logos 从零训练，不需要 GRAM 的变分训练基础设施

---

## 1. 核心命题

**多轨迹并行决策能否在端侧预算内完成？**

**核心论证**：详见 [logos-whitepaper.md §2.3](./logos-whitepaper.md#23-模块-b多轨迹并行推理端侧化关键)（多轨迹并行推理）。

**三种候选实现**：

| 实验 | 配置 | 端侧可行性 |
|------|------|----------|
| **C.1 多轨迹并行基础** | K=2（v1.0 直接用）| ✅ 已可行 |
| **C.2 多轨迹 + Radix Cache** | N 路径并行 + 共享前缀 | ✅ 延迟 ≈ 1.5x K=2 |
| **C.3 端侧多路径 Demo** | 多种并行策略组合 | ✅ 覆盖所有场景 |

---

## 2. 设计目标

**找到 Logos 端侧多假设决策的最优实现**——Radix Cache 多路径应优于单路径。

**验收标准**：
- C.2 Radix Cache 4 路径延迟 < 1.5x K=2
- C.2 多路径决策质量比单轨迹提升 ≥ 5%
- 测试时 scaling 曲线：N 路径 vs 精度的 Pareto

---

## 3. 多轨迹并行的基础

### 3.1 多路径的思想（独立实现）

> 📌 **不集成 GRAM**：我们的多路径并行**完全独立实现**——不用变分 ELBO，不用 μ, σ 可学习。

**核心机制**：
- 每条路径是一个独立的推理轨迹
- 路径间通过 **Per-Token 早退**和 **Radix Cache 共享前缀**实现端侧友好
- 综合多条路径的输出（平均 / 投票 / 加权）

```python
def multipath_reasoning(x, n_paths=4, K=2):
    """多路径推理（独立实现，不集成 GRAM）"""
    # Prefill 共享前缀
    base_h = prefill(x)
    
    # N 条独立路径（每条路径独立采样）
    paths = []
    for i in range(n_paths):
        h_i = base_h.clone()
        for k in range(K):
            # 每条路径独立推理（无 ε 注入，无变分训练）
            h_i = hrm_block(h_i, x)
        paths.append(h_i)
    
    # 综合 N 条路径
    return aggregate(paths)  # 平均 / 投票 / 加权
```

### 3.2 与 GRAM 思想的对比

| 维度 | GRAM 论文 | Logos 多路径（独立）|
|------|---------|----------|
| **多路径生成** | μ, σ 可学习 + ε ~ N(μ, σ²I) | 直接多次推理，无随机注入 |
| **变分训练** | ELBO loss + KL 散度 | 不使用 ELBO，直接回归目标 |
| **依赖训练基础设施** | 需要变分训练 | 不需要，标准 CE loss |
| **从零训练** | ✅ | ✅（更简单）|
| **是否集成 GRAM** | — | ❌ 仅作架构参考 |

**关键差异**：Logos 多路径**不依赖变分训练**，可以直接复用 Logos H/L 标准训练的推理流程。**这是从零架构的严肃承诺**。

---

## 4. Radix Cache 多路径集成（C.2）

### 4.1 完整架构

```python
class RadixCache:
    """Radix Tree 节点（共享前缀缓存）"""
    def __init__(self, hidden_state=None):
        self.hidden_state = hidden_state
        self.children = {}  # {prefix_hash: RadixTreeNode}

class RadixCacheMultipath(nn.Module):
    """Radix Cache 多路径并行"""
    def __init__(self, base_model, n_paths=4, K=2):
        super().__init__()
        self.base_model = base_model  # v1.0 A.3 混合风格
        self.n_paths = n_paths
        self.K = K
    
    def forward(self, input_ids):
        # Prefill 共享前缀
        base_h = self.base_model.embed(input_ids)
        radix_cache = RadixCache(base_h)
        
        # N 条路径并行采样
        paths = []
        for i in range(self.n_paths):
            h_i = base_h.clone()
            
            for k in range(self.K):
                # 检索 Radix Cache
                cached = radix_cache.get(h_i.hash())
                if cached is not None:
                    h_i = cached
                    continue
                
                # 标准推理（无 ε 注入）
                h_i = self.base_model.hrm_step(h_i, input_ids)
                radix_cache.put(h_i.hash(), h_i)
            
            paths.append(h_i)
        
        # 综合 N 条路径
        # 策略 1：平均
        # h_final = torch.stack(paths).mean(dim=0)
        
        # 策略 2：投票（仅对离散输出）
        # h_final = vote(paths)
        
        # 策略 3：加权（按置信度）
        h_final = weighted_aggregate(paths)
        
        return self.base_model.lm_head(h_final)
```

### 4.2 关键优化

| 优化 | 收益 |
|------|------|
| **路径并行**：N 条轨迹在同一 forward 内并行 | 延迟 × N → 延迟 × 1 |
| **Radix Cache**：共享前缀不重复计算 | 算力 × 0.3-0.5 |
| **预计算路径**：提前探索常见路径模式 | 命中率 50%+ |

---

## 5. 层次化集成（C.3）

### 5.1 完整架构

```python
class HierarchicalMultipath(nn.Module):
    """层次化推理 + 多路径"""
    def __init__(self, base_model, K_main=2, K_sub=1, n_subpaths=3):
        super().__init__()
        self.base_model = base_model
        self.K_main = K_main
        self.K_sub = K_sub
        self.n_subpaths = n_subpaths
    
    def forward(self, input_ids):
        h = self.base_model.embed(input_ids)
        
        # 主推理第一阶段（K_main 步）
        for k in range(self.K_main):
            h = self.base_model.hrm_step(h, input_ids)
        
        # 触发子推理（并行 n_subpaths 条）
        sub_results = []
        for i in range(self.n_subpaths):
            h_sub = h.clone()
            for k in range(self.K_sub):
                h_sub = self.base_model.hrm_step(h_sub, input_ids)
            sub_results.append(h_sub)
        
        # 主推理第二阶段（融合子结果）
        h = self.base_model.hrm_step_with_context(h, input_ids, sub_results)
        
        return self.base_model.lm_head(h)
```

### 5.2 与 C.2 对比

| 维度 | C.2 Radix Cache | C.3 层次化 |
|------|----------------|------------|
| **延迟** | ≈ 1.5x K=2 | ≈ 2-3x K=2 |
| **决策质量** | 多假设综合 | 主 + 子推理融合 |
| **适用场景** | 多假设决策 | 可分解推理 |
| **实现复杂度** | 🟡 中等 | 🔴 高（需主/子 block 设计）|

---

## 6. 测试时 Scaling 实验

### 6.1 Scaling 曲线设计

| 配置 | N 路径 | K 步 | 总循环步 | 延迟 |
|------|:---:|:---:|:---:|:---:|
| 基准 | 1 | 2 | 2 | 100ms |
| 多路径 | 2 | 2 | 4 | ~120ms |
| 多路径 | 4 | 2 | 8 | ~130ms |
| 多路径 | 8 | 2 | 16 | ~170ms |
| 多路径 + 早退 | 4 | 平均 3 | 12 | ~150ms |
| 多路径 + 层次化 | 4 | 主+子 | ~12 | ~180ms |

### 6.2 绘制 Pareto 曲线

```
精度
 ↑
 │  ●  8 路径 K=2（~170ms）
 │  ●
 │   ●  4 路径 K=2（~130ms）
 │    ●
 │     ●  4 路径 + 早退（~150ms）
 │      ●
 │       ●  2 路径 K=2（~120ms）
 │        ●
 │         ●  1 路径 K=2（100ms）
 │          ●  1 路径 K=1（50ms）
 └────────────────────────────→ 延迟
```

### 6.3 推荐配置

基于 Pareto 曲线：
- **端侧对话**：1 路径 K=2（100ms）+ 早退
- **端侧决策**：4 路径 K=2（~130ms）
- **离线规划**：8 路径 K=2（~170ms）或 1 路径 K=8 PLT（~120ms）

---

## 7. 实验设计

### 7.1 训练配置

```python
# 共享训练配置（基于 v2.0 推荐组合）
training_config = {
    'base': 'logos_v2_recommended',  # A.3 + 多种循环策略
    'lr': 1e-4,
    'warmup': 2000,
    'batch_size': 64,
    'seq_length': 2048,
    'total_tokens': 4_000_000_000,
}
```

### 7.2 评估指标

| 指标 | 目的 |
|------|------|
| **多假设决策精度** | vs 单轨迹基线 |
| **N 路径 scaling 曲线** | 端侧最优 N |
| **Radix Cache 命中率** | 缓存效率 |
| **单 token 延迟** | 端侧预算 |

### 7.3 决策任务数据集

| 数据集 | 类型 | 用途 |
|--------|------|------|
| **nuScenes mini** | 自动驾驶多假设 | 主要任务 |
| **GSM8K 多解** | 数学多解推理 | 数学决策 |
| **ARC-AGI multi-path** | 抽象推理多路径 | 通用推理 |

---

## 8. 验收标准

### 8.1 单实验验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| 多路径基础训练稳定 | loss 收敛 | 收敛且无 mode collapse |
| C.2 Radix Cache 4 路径延迟 | < 1.5x K=2 | < 1.2x K=2 |
| C.2 Radix Cache 4 路径决策质量 | 比单轨迹 +5% | 比单轨迹 +10% |
| C.3 层次化延迟 | < 3x K=1 | < 2.5x K=1 |
| C.3 层次化决策质量 | 比单轨迹 +3% | 比单轨迹 +8% |

### 8.2 测试时 Scaling 验收

| 验收项 | 合格线 |
|-------|:---:|
| N 路径 vs 精度 | 单调不减 |
| 饱和点 | N=4 或 N=8 明显饱和 |
| Pareto 曲线 | 推荐配置清晰 |

---

## 9. 决策传递：64M → 300M

### 9.1 64M 出《验证报告》后

| 64M 结论 | 300M 训练策略 |
|---------|----------|
| Radix Cache N=4 路径最优 | 300M 集成 N=4 Radix Cache（**完全从零训练**）|
| 层次化最优 | 300M 集成层次化 |
| 单路径多轨迹已足够 | 300M 用单路径（但有并行化）|
| Radix Cache 实现复杂 | 退回单路径 |

### 9.2 推荐组合（最优假设）

**300M 训练启动（Month 3+）**：
- A.3 混合风格基座
- Per-Token 早退
- Radix Cache N=4 路径（如果验证有效）

**关键**：300M **不加载 64M 权重**——完全从零训练。

---

## 10. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-64m-validation-plan.md §5](./logos-64m-validation-plan.md#5-v30多轨迹并行决策) | 本文档的父级 |
| [logos-v1-architecture.md](./logos-v1-architecture.md) | v1.0 双时间尺度对比（v3.0 基座）|
| [logos-v2-architecture.md](./logos-v2-architecture.md) | v2.0 多种循环策略（v3.0 推荐组合来源）|
| [logos-whitepaper.md §2.3](./logos-whitepaper.md#23-模块-b多轨迹并行推理端侧化关键) | 多轨迹并行高层架构 |
| [logos-k-strategy.md §2.3](./logos-k-strategy.md#23-radix-cache-多路径并行) | Radix Cache 详细论证 |
| [docs/references/gram.md](../references/gram.md) | GRAM 原论文笔记（仅参考）|
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | PLT 架构（Radix Cache 借鉴）|

---

**最后更新**：2026-07-29（v1.2 重大调整：去除 GRAM 集成）
**作者**：来自工作流（Logos v3.0 架构设计）
**版本**：v1.2