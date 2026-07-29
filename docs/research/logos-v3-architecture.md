# Logos-Native-64M v3.0 架构设计：GRAM + Radix Cache 多路径决策

> **一句话定位**：在 v2.0 最优循环策略（K=2 + 早退 + Radix Cache 三件套应最优）上集成 GRAM 多轨迹——通过 Radix Cache 多路径并行让 N 条轨迹在端侧预算内完成，绘制"路径数 × 决策质量" scaling 曲线。
> **上游文档**：[logos-64m-validation-plan.md §4](./logos-64m-validation-plan.md#4-v30gram--radix-cache-多路径决策)
> **下游文档**：交付《Logos 64M 验证报告》→ 启动 HRM-Text 1B 集成
> **最后更新**：2026-07-29

---

## 0. 核心命题

**GRAM 多轨迹决策能否在端侧预算内完成？**

**核心论证**：详见 [logos-whitepaper.md §2.3](./logos-whitepaper.md#23-模块-bgram-多轨迹扰动)（Radix Cache 多路径并行）。

**三种候选实现**：

| 实验 | 配置 | 端侧可行性 |
|------|------|----------|
| **C.1 GRAM 单轨迹基线** | K=2 + ε 注入（v1.0 直接用）| ✅ 已可行 |
| **C.2 GRAM + Radix Cache** | 4 路径并行 + 共享前缀 | ✅ 延迟 ≈ 1.5x K=2 |
| **C.3 GRAM + 层次化** | 主推理 + 子推理并行 | ✅ 延迟 ≈ 2-3x K=2 |

---

## 1. 设计目标

**找到 Logos 端侧多假设决策的最优实现**——Radix Cache 多路径应优于层次化。

**验收标准**：
- C.2 Radix Cache 4 路径延迟 < 1.5x K=2
- C.2 多路径决策质量比单轨迹提升 ≥ 5%
- 测试时 scaling 曲线：N 路径单调增，找到饱和点

---

## 2. GRAM 多轨迹基础

### 2.1 GRAM 变分注入

**核心机制**：在 H module residual 后注入 ε ~ N(μ, σ²I)，变分 ELBO 训练。

```python
def gram_h_module(z_H):
    """H module with GRAM ε injection"""
    z_H_new = H_module(z_H)
    
    # GRAM 变分注入（训练时）
    if self.training:
        mu = self.mu_proj(z_H_new)        # 学习均值
        sigma = self.sigma_proj(z_H_new)  # 学习方差
        eps = torch.randn_like(z_H_new) * sigma + mu
        z_H_new = z_H_new + eps
    
    return z_H_new
```

### 2.2 训练时变分 ELBO

```python
def gram_loss(z_H_pred, target_ids, mu, sigma):
    """GRAM 变分损失"""
    # 重建损失
    ce_loss = cross_entropy(z_H_pred, target_ids)
    
    # KL 散度（约束 μ, σ）
    kl_loss = -0.5 * torch.mean(1 + torch.log(sigma**2) - mu**2 - sigma**2)
    
    return ce_loss + 0.1 * kl_loss  # β-VAE 风格
```

### 2.3 推理时多轨迹

```python
def gram_inference(x, n_trajectories=4):
    """GRAM 推理：采样多条轨迹并综合"""
    base_h = prefill(x)
    
    trajectories = []
    for i in range(n_trajectories):
        # 独立采样 ε（每条轨迹独立）
        torch.manual_seed(i)  # 或保留随机性
        h_i = gram_h_module_chain(base_h)  # K 步推理
        trajectories.append(h_i)
    
    # 综合 N 条轨迹
    return aggregate(trajectories)  # 平均 / 投票 / 加权
```

---

## 3. Radix Cache 多路径集成（C.2）

### 3.1 完整架构

```python
class RadixCacheGRAM(nn.Module):
    """Radix Cache + GRAM 多路径并行"""
    def __init__(self, base_model, n_paths=4, K=2):
        super().__init__()
        self.base_model = base_model  # v1.0 A.3 混合风格
        self.n_paths = n_paths
        self.K = K
    
    def forward(self, input_ids):
        # Prefill 共享前缀
        base_h = self.base_model.embed(input_ids)
        radix_cache = RadixCache(base_h)
        
        # N 条轨迹并行采样
        trajectories = []
        for i in range(self.n_paths):
            h_i = base_h.clone()
            
            for k in range(self.K):
                # 检索 Radix Cache
                cached = radix_cache.get(h_i.hash())
                if cached is not None:
                    h_i = cached
                    continue
                
                # GRAM 变分推理
                h_i = self.base_model.hrm_step(h_i, input_ids)
                radix_cache.put(h_i.hash(), h_i)
            
            trajectories.append(h_i)
        
        # 综合 N 条轨迹
        # 策略 1：平均
        # h_final = torch.stack(trajectories).mean(dim=0)
        
        # 策略 2：投票（仅对离散输出）
        # h_final = vote(trajectories)
        
        # 策略 3：加权（按置信度）
        h_final = weighted_aggregate(trajectories)
        
        return self.base_model.lm_head(h_final)
```

### 3.2 Radix Cache 实现

```python
class RadixTreeNode:
    """Radix Tree 节点（共享前缀缓存）"""
    def __init__(self, hidden_state=None):
        self.hidden_state = hidden_state
        self.children = {}  # {prefix_hash: RadixTreeNode}

class RadixCache:
    def __init__(self, base_hidden):
        self.root = RadixTreeNode(base_hidden)
    
    def get(self, prefix_hash):
        """检索缓存的前缀 hidden state"""
        node = self.root
        for h in prefix_hash:
            if h not in node.children:
                return None
            node = node.children[h]
        return node.hidden_state
    
    def put(self, prefix_hash, hidden_state):
        """存储前缀 hidden state"""
        node = self.root
        for h in prefix_hash:
            if h not in node.children:
                node.children[h] = RadixTreeNode()
            node = node.children[h]
        node.hidden_state = hidden_state
```

### 3.3 关键优化

| 优化 | 收益 |
|------|------|
| **路径并行**：N 条轨迹在同一 forward 内并行 | 延迟 × N → 延迟 × 1 |
| **Radix Cache**：共享前缀不重复计算 | 算力 × 0.3-0.5 |
| **预计算路径**：提前探索常见路径模式 | 命中率 50%+ |

---

## 4. 层次化 GRAM 集成（C.3）

### 4.1 完整架构

```python
class HierarchicalGRAM(nn.Module):
    """层次化推理 + GRAM 多轨迹"""
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

### 4.2 与 C.2 对比

| 维度 | C.2 Radix Cache | C.3 层次化 |
|------|----------------|------------|
| **延迟** | ≈ 1.5x K=2 | ≈ 2-3x K=2 |
| **决策质量** | 多假设综合 | 主 + 子推理融合 |
| **适用场景** | 多假设决策 | 可分解推理 |
| **实现复杂度** | 🟡 中等 | 🔴 高（需主/子 block 设计）|

---

## 5. 测试时 Scaling 实验

### 5.1 Scaling 曲线设计

| 配置 | N 路径 | K 步 | 总循环步 | 延迟 |
|------|:---:|:---:|:---:|:---:|
| 基准 | 1 | 2 | 2 | 100ms |
| 多路径 | 2 | 2 | 4 | ~110ms |
| 多路径 | 4 | 2 | 8 | ~130ms |
| 多路径 | 8 | 2 | 16 | ~170ms |
| 多路径 + 早退 | 4 | 平均 3 | 12 | ~150ms |
| 多路径 + 层次化 | 4 | 主+子 | ~12 | ~180ms |

### 5.2 绘制 Pareto 曲线

```
精度
 ↑
 │  ●  8 路径 K=2（~170ms）
 │  ●
 │   ●  4 路径 K=2（~130ms）
 │    ●
 │     ●  4 路径 + 早退（~150ms）
 │      ●
 │       ●  2 路径 K=2（~110ms）
 │        ●
 │         ●  1 路径 K=2（100ms）
 │          ●  1 路径 K=1（50ms）
 └────────────────────────────→ 延迟
```

### 5.3 推荐配置

基于 Pareto 曲线：
- **端侧对话**：1 路径 K=2（100ms）+ 早退
- **端侧决策**：4 路径 K=2（~130ms）
- **离线规划**：8 路径 K=2（~170ms）或 1 路径 K=8 PLT（~120ms）

---

## 6. 实验设计

### 6.1 训练配置

```python
# 共享训练配置（基于 v2.0 推荐组合）
training_config = {
    'base': 'logos_v2_recommended',  # K=2 + 早退 + Radix Cache
    'lr': 1e-4,
    'warmup': 2000,
    'batch_size': 64,
    'seq_length': 2048,
    'total_tokens': 4_000_000_000,
    
    'gram': {
        'mu_proj_hidden': 768,
        'sigma_proj_hidden': 768,
        'kl_weight': 0.1,
    },
}
```

### 6.2 评估指标

| 指标 | 目的 |
|------|------|
| **多假设决策精度** | vs 单轨迹基线 |
| **N 路径 scaling 曲线** | 端侧最优 N |
| **Radix Cache 命中率** | 缓存效率 |
| **单 token 延迟** | 端侧预算 |
| **KL 散度监控** | GRAM 变分稳定性 |

### 6.3 决策任务数据集

| 数据集 | 类型 | 用途 |
|--------|------|------|
| **nuScenes mini** | 自动驾驶多假设 | 主要任务 |
| **GSM8K 多解** | 数学多解推理 | 数学决策 |
| **ARC-AGI multi-path** | 抽象推理多路径 | 通用推理 |

---

## 7. 验收标准

### 7.1 单实验验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| C.1 GRAM 训练稳定 | ELBO 收敛 | 无 mode collapse |
| C.2 Radix Cache 4 路径延迟 | < 1.5x K=2 | < 1.2x K=2 |
| C.2 Radix Cache 4 路径决策质量 | 比单轨迹 +5% | 比单轨迹 +10% |
| C.3 层次化延迟 | < 3x K=1 | < 2.5x K=1 |
| C.3 层次化决策质量 | 比单轨迹 +3% | 比单轨迹 +8% |

### 7.2 测试时 Scaling 验收

| 验收项 | 合格线 |
|-------|:---:|
| N 路径 vs 精度 | 单调不减 |
| 饱和点 | N=4 或 N=8 明显饱和 |
| Pareto 曲线 | 推荐配置清晰 |

---

## 8. 决策传递：64M → 1B

### 8.1 64M 出《验证报告》后

| 64M 结论 | 1B 集成策略 |
|---------|-----------|
| Radix Cache N=4 路径最优 | 1B v1.5 集成 N=4 Radix Cache |
| 层次化最优 | 1B v1.5 集成层次化 |
| 单轨迹 GRAM 已足够 | 1B v1.5 用单轨迹 GRAM |
| Radix Cache 实现复杂 | 退回单轨迹 GRAM |

### 8.2 推荐组合（最优假设）

**v1.0 1B 集成（Month 1-5）**：
- A.3 混合风格基座
- K=2 默认
- Per-Token 早退

**v1.5 1B 集成（Month 7-10）**：
- 上述 + Radix Cache N=4 路径
- GRAM 变分注入

**v2.0 完整融合（Month 12+）**：
- 上述 + SADKO ELF Memory KV → HRM Cross-Attention

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-64m-validation-plan.md §4](./logos-64m-validation-plan.md#4-v30gram--radix-cache-多路径决策) | 本文档的父级 |
| [logos-v1-architecture.md](./logos-v1-architecture.md) | v1.0 双时间尺度对比（v3.0 基座）|
| [logos-v2-architecture.md](./logos-v2-architecture.md) | v2.0 多种循环策略（v3.0 推荐组合来源）|
| [logos-whitepaper.md §2.3](./logos-whitepaper.md#23-模块-bgram-多轨迹扰动) | GRAM + Radix Cache 高层架构 |
| [logos-k-strategy.md §4.3](./logos-k-strategy.md#43-radix-cache-多路径并行你提出的方案) | Radix Cache 详细论证 |
| [docs/references/gram.md](../references/gram.md) | GRAM 原论文笔记 |
| [docs/references/loopcoder-v2.md](../references/loopcoder-v2.md) | PLT 架构（Radix Cache 借鉴）|

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos v3.0 架构设计）
**版本**：v1.0