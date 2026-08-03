# Logos 消费 Hippo FSQ 的条件设计

> **一句话定位**：Logos H/L 循环通过 FSQ 离散通道消费 Hippo 记忆的条件架构设计——**仅当**前置条件全部满足后激活
> **性质**：条件设计文档（conditional design，**不是**立即实施计划）
> **最后更新**：2026-07-31
> **触发动机**：DiscoLoop 调研（arXiv:2607.00341）+ Oracle 评审建议
> **上游约束**：[codebook-config-sop.md](./codebook-config-sop.md)（前置依赖 1） + [64m-validation-plan.md §3.5](./64m-validation-plan.md#35-表征对齐探针新增oracle-建议)（前置依赖 2）

---

## 0. 关键定位：本文档不是立即实施计划

> ⚠️ **必须先读**：本文档描述 Logos 如何**消费 Hippo FSQ 码本作为 DiscoLoop 式离散通道**。**该机制仅在以下前置条件全部满足后才激活**：
>
> **前置条件 1（配置层）**：HIPPO FSQ 配置统一为 `[8,8,4] = 256`——见 [codebook-config-sop.md](./codebook-config-sop.md)
>
> **前置条件 2（机制层）**：Hippo 单独验证通过——见 [hippo/README.md §3 验证哲学](../hippo/README.md#3-验证哲学继承-sadko-四)（SADKO 64M 实验 A：码本使用率 > 80%）
>
> **前置条件 3（表征层）**：Logos H/L baseline 确实存在表征错位——见 [64m-validation-plan.md §3.5 A/B 实验](./64m-validation-plan.md#35-表征对齐探针新增oracle-建议)
>
> **前置条件 4（接口层）**：Hippo 胼胝体契约达到 T+2（reference code 阶段）——见 [hippo/README.md §2.4](../hippo/README.md#24-演进路径)

**任一条件未达成 → 本文档的机制不启动，Logos 维持独立 H/L 循环架构，不引入 FSQ 消费**。

---

## 1. 设计动机

### 1.1 来自 DiscoLoop 的启发

DiscoLoop 证明：循环 Transformer 在多跳推理 OOD 上的失败可通过**离散嵌入通道**修复。原始论文用 LM 词表（vocab）的 soft 期望实现：

$$
\Phi_{\text{LM}}(h) = \sum_v p_v(h)\, W[v]
$$

### 1.2 用 Hippo FSQ 替代 LM 词表的潜在优势

| 维度 | LM 词表（DiscoLoop 原方案）| Hippo FSQ（潜在替代）|
|------|-------------------------|---------------------|
| 通道基数 | vocab_size ≈ 6400 | codebook_size = 256 |
| 通道语义 | 词级 token（语言学单元）| chunk 级语义锚点（流形几何）|
| 多模态适配 | ❌ 语言专属 | ✅ 模态无关 |
| 端侧激活内存 | `O(T × 6400 × d)` | `O(T × 256 × d)` ≈ 25× 更小 |
| 与 Hippo 记忆底座协同 | ❌ 独立 | ✅ 复用已训练 KV |

### 1.3 Oracle 警示（必须正视）

| 警示 | 详细 | 缓解 |
|------|------|------|
| **FSQ ≠ LM vocab 通道（语义层）** | DiscoLoop 用 LM vocab 描述的是 token-level 桥接实体；FSQ 描述的是 pooled chunk 全局记忆 | Logos 必须先验证 "FSQ 通道可消费性"，见 §4 验证实验 |
| **码本冻结约束（INV-2）** | Hippo 胼胝体契约 INV-2：`idx → embedding` 映射对左脑只读冻结 | Logos 侧 adapter 可训练；idx 选择由 Hippo 输出决定 |
| **维度匹配需验证** | FSQ `code_embed = Embedding(256, 768)` 与 Logos hidden size 768 匹配；若不同需显式投影 | 在 §3 接口契约中明确 |
| **训练/推理一致性** | Hippo 训练时 FSQ 量化可能用 STE 或 soft relaxation，与 Logos 消费侧需一致 | §3.2 训练契约明确 |

---

## 2. 架构设计

### 2.1 整体公式

**完整 Logos H/L 循环（含 FSQ 通道）**：

```python
# 初始化
h_L = z_L_init.expand_as(h_H)
h_H = embed(input_ids) * embedding_scale

for h_cycle in range(H_cycles):  # 默认 2
    for l_cycle in range(L_cycles):  # 默认 3
        # === 现有 Logos H/L 循环 ===
        h_L = L_module(h_L + h_H)

    # === FSQ 离散通道（仅 H 边界注入）===
    if FSQ_CONSUMPTION_ENABLED:  # 前置条件全部满足
        # 1. 用 LM head 解码 h_H，得到 soft distribution over Hippo codebook
        #    （注意：这里用 Logos 自己的 LM head，不是 Hippo 的）
        p_code = softmax(LM_head(h_H) / tau, dim=-1)  # [batch, seq, 256]
        #    注意：LM_head 输出维度需重映射到 256（非 vocab_size）
        #    见 §3.2 维度匹配方案 B

        # 2. 期望得到 FSQ 嵌入向量
        phi_h = einsum("bsc,cd->bsd", p_code, hippo_code_embed.weight)  # [batch, seq, d]
        phi_h = rms_norm(phi_h)

        # 3. Per-token 门控注入
        alpha = sigmoid(gate_net(h_H))  # [batch, seq, 1]  (d+1 个门控参数)
        h_H = H_module(h_H + alpha * phi_h)
    else:
        h_H = H_module(h_H + h_L)

return h_H
```

### 2.2 与原始 Logos H/L 循环的差异

| 项 | 原始 Logos（[whitepaper §2.2](./whitepaper.md#22-模块-blogos-分层递归潜空间引擎-750m)）| 含 FSQ 通道 |
|---|-------------------------------------|------------|
| H 循环输入 | `h_H + h_L` | `h_H + h_L + α · Φ_FSQ(h_H)` |
| 通道数 | 2（连续 + 加法注入）| 3（连续 + 加法注入 + 离散 FSQ 通道） |
| 参数增量 | 0（基线）| + (256 × d) + (d + 1) ≈ 197K（d=768 时）|
| 计算开销 | 基线 | + O(H × T × 256 × d) per H 边界 |

### 2.3 端侧可行性论证（待前置条件 3 验证）

| 维度 | 估算 | 与 K>4 端侧化的兼容性 |
|------|------|---------------------|
| 通道基数 256（vs LM vocab 6400）| 25× 更小 | ✅ 更利于端侧 |
| 激活内存（per H 边界）| `T × 256 × 768 × 2 bytes = 393KB × T/100` | ✅ T=512 时仅 2MB |
| 训练 step 增加 | ≤ 5% | ✅ |
| 推理延迟（per H 边界）| + ~3ms（K=2 时） | ⚠️ K=8 时 +12ms 可能不可行 |

**Oracle 警示**：K=8 + FSQ + 端侧需独立 latency 测量，未在 DiscoLoop 论文中验证。

---

## 3. 接口契约

### 3.1 与 Hippo 胼胝体契约的关系

**已有契约**（[hippo/README.md §2.2](../hippo/README.md#22-接口契约五元组骨架)）：

| Hippo 契约维度 | 现有 | Logos 消费 FSQ 时的新增 |
|----------------|------|------------------------|
| **I1 Memory KV** | Logos Cross-Attention 读取 | 不变 |
| **I2 Codebook Protocol** | FSQ 索引序列 `idx ∈ ℕ^L` | **本设计核心**：Logos 直接消费 idx 序列 |
| **I3 Retrieval API** | Top-K 码字 + KV + 置信度 | 不变 |
| **I4 Incremental Update** | 训练时更新 | 不变 |
| **I5 Failure Fallback** | 静默降级 | **新增不变量 INV-6**（见下）|

### 3.2 Logos 侧新增强制约束（待 Hippo 协调签字）

| # | 不变量 | 描述 |
|---|------|------|
| **INV-6** | FSQ 通道默认关闭 | 与 Hippo INV-5 一致，Logos 内部同样要求 FSQ_CONSUMPTION_ENABLED 标志默认 False |
| **INV-7** | Gate=0 时退化为原始 Logos | 若 α gate 输出为 0，本设计必须等价于不含 FSQ 通道的 H/L 循环（Oracle 强调的"安全降级"）|
| **INV-8** | 维度匹配显式化 | Logos hidden size `d_L` 与 Hippo code_embed dim `d_H` 必须显式映射（投影 or 直接相等）|
| **INV-9** | 训练/推理一致 | Logos 训练时使用的 FSQ relaxation（soft expectation）必须与 Hippo 训练时的 FSQ 实现匹配——不允许 train-soft/infer-hard 不一致 |

### 3.3 维度匹配方案

**方案 A（推荐）**：Logos hidden size 与 Hippo code_embed dim 直接相等（如均为 768）
- ✅ 零额外投影参数
- ✅ 实现最简
- ⚠️ 需 Logos 选定 hidden size = 768（当前 v1-arch §3 设计如此）

**方案 B**：不等时显式投影
- Logos 侧加 `proj = Linear(d_L, d_H)`，再 `phi_h = proj(phi_h)`
- ⚠️ 增加参数与计算开销
- 🟡 仅当方案 A 受限（如 Hippo 改为 d_H=512）时启用

---

## 4. 验证实验（前置条件全部满足后启动）

### 4.1 实验列表

| 实验 | 名称 | 目的 | 通过标准 | 周期 |
|------|------|------|:---:|:---:|
| **E.1** | FSQ 通道可消费性（FSQ-vs-LM-vocab）| 验证 FSQ 通道效果不劣于 LM vocab 通道 | OOD 多跳：C_FSQ ≥ 0.9 × C_LM | 1 周 |
| **E.2** | 端侧延迟预算 | 验证 K=2 + FSQ + 端侧总延迟 ≤ 200ms | 单 token ≤ 200ms | 3 天 |
| **E.3** | Gate 行为分析 | 验证 α gate 不坍缩到 0/1 | α 分布熵 > 1.0 | 2 天 |
| **E.4** | 量化兼容性 | 验证 INT8 推理下通道不退化 | top-token 翻转率 < 20% | 1 周 |

### 4.2 E.1 实验设计（最关键）

**对比组**：
- C1：原始 Logos（无 Φ 通道）—— 基线
- C2：Logos + LM-vocab Φ（H 边界）—— DiscoLoop 原方案
- C3：Logos + FSQ Φ（H 边界）—— 本设计
- C4：Logos + Shuffled FSQ Φ（对照）

**数据集**：entity-disjoint 2-hop + 3-hop 任务（同 §3.5 探针数据集）

**关键判定**：
- C3 ≥ C2 → FSQ 通道可行，启动本设计
- C3 ≈ C1 → FSQ 通道无价值（Oracle 警示：应保守不引入）
- C3 < C1 → 引入破坏 Logos，**回退本设计**

### 4.3 实验时序

```
Phase 0 [codebook-config-sop.md] 通过（前置条件 1）
   ↓
Phase 1 v3.0 [Hippo 实验 A] 通过：码本使用率 > 80%（前置条件 2）
   ↓
§3.5 探针 实验 A+B：确认表征错位存在（前置条件 3）
   ↓
Hippo 胼胝体契约达到 T+2：reference code 可用（前置条件 4）
   ↓
启动 E.1-E.4 验证
   ├─ E.1 失败 → 回退本设计，不引入 FSQ 通道
   └─ E.1 通过 → 进入 E.2-E.3-E.4
       └─ E.2 失败（端侧不可行）→ 推迟到 300M 升格（INT8 通道量化放宽）
       └─ 全部通过 → 启动 Logos v1.4+ 引入 FSQ 通道
```

---

## 5. 与现有 Logos 机制的关系

### 5.1 Nano-WM Gate（必须独立）

**严格隔离**：
- Nano-WM Gate 控制**外部知识**（Hippo Memory KV）注入
- FSQ 通道 Gate 控制**离散锚点**（自解码再注入）注入
- **两者不共享门控**——Oracle 明确要求 2×2 消融

```
输出 = main_ffn(x) + nano_gate * parallel_ffn(nano_knowledge)
                  + fsq_alpha * phi_fsq(h)            # 新增，与 nano_gate 独立
```

### 5.2 Per-Token 早退

**新增收敛信号**（Oracle 警示）：
- 现有：只看 hidden state delta
- 新增：top-token 持久度 / Jensen-Shannon 散度跨步变化
- **原因**：FSQ snapping 可能制造虚假收敛

### 5.3 多轨迹决策（v3.0）

**风险**：强离散通道可能把多条轨迹拉向同一 code → 摧毁 v3.0 多轨迹设计
- **缓解**：轨迹独立时 alpha gate 应学得低（FSQ 通道选择性弱化）

---

## 6. 风险与回退

| 风险 | 概率 | 影响 | 回退 |
|------|:---:|:---:|------|
| 前置条件不达成 | 🟠 中 | 中 | 本设计不启动，Logos 维持现状 |
| FSQ 通道效果不优于 LM vocab | 🟠 中 | 中 | 保留 LM vocab Φ 方案（[64m-validation-plan.md §3.5 实验 C](./64m-validation-plan.md)）|
| 端侧延迟超预算 | 🟡 中 | 高 | 推迟到 300M + INT8 通道放宽 |
| FSQ 与 SADKO 训练目标冲突 | 🟡 低 | 中 | 启用 INV-9 训练一致性检查 |
| Logos 与 Hippo 双向干扰（耦合后失败归因难）| 🟠 中 | 高 | **回退原则：FSQ_CONSUMPTION_ENABLED 标志关闭即可瞬时回退** |

---

## 7. 相关文档

| 文档 | 关系 |
|------|------|
| [codebook-config-sop.md](./codebook-config-sop.md) | **前置依赖 1**：FSQ 配置统一 |
| [64m-validation-plan.md §3.5](./64m-validation-plan.md) | **前置依赖 2-3**：表征对齐探针 |
| [hippo/README.md §2.4](../hippo/README.md) | **前置依赖 4**：胼胝体契约演进路径 |
| [discoloop.md §6.1](../../references/discoloop.md) | 设计灵感来源 |
| [whitepaper.md §2.2](./whitepaper.md) | Logos 原始 H/L 循环架构 |
| [fsq-consumption-design.md §6 风险](#6-风险与回退) | 与本文档同步创建（跨线协调）|

---

**最后更新**：2026-07-31
**作者**：来自 DiscoLoop 调研工作流（条件设计，**不立即实施**）
**状态**：🟡 待前置条件满足后激活