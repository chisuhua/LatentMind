# SADKO 右脑 Phase 0 执行手册：生命体征监测基线

> **一句话定位**：Phase 0（FM+FSQ 收敛验证）的执行手册——量化指标、观测频率、死亡线与逃生舱
> **关联文档**：[elf-graph-emergence.md](./elf-graph-emergence.md)（促生长/证生长理论）· [elf-vs-gdm-review.md](./elf-vs-gdm-review.md) §3.2（四阶段验证总表）
> **最后更新**：2026-07-29

Phase 0 不是"跑通代码"，而是**建立 SADKO 右脑的"生命体征监测基线"**。如果 Flow Matching 无法稳定地将 KV Cache 映射到 FSQ 码字并精确重构，后续所有关于"图结构"或"事实分离"的实验都是在拟合噪声。

---

## 一、 核心目标与验收标准

**唯一目标**：验证 `KV Cache → FM Encoder → FSQ → FM Decoder → Reconstructed KV` 这条链路在数学上收敛且可逆。

| 维度 | 核心指标 | ✅ 通过阈值 | ⚠️ 警戒区 | ❌ 死亡线 |
| :--- | :--- | :--- | :--- | :--- |
| **训练稳定性** | Velocity Loss（$\mathcal{L}_{vel}$） | 单调下降，1k 步内方差 < 5% | 震荡但均值下降 | 发散 / NaN / 3k 步无下降 |
| **码本健康度** | FSQ Codebook Utilization | > 90%（活跃码字比例） | 70% - 90% | < 70%（码本坍塌） |
| **重构精度** | KV Reconstruction MSE | < 基准值（见下） | 基准值 ~ 1.5x | > 2x 基准值 |
| **语义保持度** | Probe Accuracy（线性探针） | > 85%（实体/属性分类） | 70% - 85% | < 70% |
| **ODE 一致性** | Forward-Backward Cycle Error | < 1e-3（归一化 L2） | 1e-3 ~ 1e-2 | > 1e-2 |

> **📊 "基准值"如何获取**：正式训练前，用一个**纯 Autoencoder（无 FM、无 FSQ）**跑同样的 KV 压缩任务 1k 步，记录其 MSE 作为"理论下界"。SADKO 的 MSE 允许比此高 20-50%（量化和流匹配有损），但不能高出数倍。

---

## 二、 执行步骤

### Step 1: 数据准备与小规模验证（Day 1-2）

- **数据集**：10K-50K 条高质量百科/文档样本。**不要使用全量数据**——Phase 0 的目的是调试，不是训练模型。
- **预处理**：用冻结的主干 LLM 提取 KV Cache，做 LayerNorm 或 PCA 白化，确保各层/各头数值范围一致。**未归一化的 KV 是 FM 训练崩溃的首要原因。**
- **Sanity Check**：先用 1K 样本跑 100 步，确认 Loss 不为 NaN、FSQ 有梯度、GPU 显存未溢出。

### Step 2: 高频监控仪表盘搭建（Day 2）

不要只看 TensorBoard 的 Loss 曲线，必须搭建实时看板：

```python
# 伪代码：Phase 0 核心监控项
metrics = {
    "velocity_loss": mean(||v_pred - v_target||^2),
    "codebook_usage": unique(fsq_indices) / total_codes,
    "recon_mse": mean(||kv_recon - kv_orig||^2),
    "fsq_entropy": -sum(p * log(p)),        # 码字分布熵，越高越均匀
    "ode_cycle_err": mean(||encode(decode(z)) - z||),
    "grad_norm_flow": norm(grads_flow_params),
    "grad_norm_fsq": norm(grads_fsq_params),
}
```

- **采样频率**：每 50 步记录一次。Phase 0 的崩溃往往发生在前 500 步。
- **可视化**：每 500 步对 FSQ 码字做 UMAP 投影。观察是否从"一团混沌"逐渐形成"离散簇"（因子解耦的密码子表结构，而非完整图拓扑——见 [graph-emergence](./elf-graph-emergence.md) §3.2）。若始终均匀一片或几个孤立点，立即干预。

### Step 3: 分阶段训练与干预策略（Day 3-14）

| 阶段 | 步数 | 关注重点 | 数据异常时的干预动作 |
| :--- | :--- | :--- | :--- |
| **Warmup** | 0-1k | Grad Norm, Loss 初始值 | Grad Norm > 10 → 降低 LR / 增加 Warmup；Loss 不降 → 检查 KV 归一化 / ODE Solver 配置 |
| **Codebook Awakening** | 1k-3k | FSQ Entropy, Usage | Entropy 持续走低 → 增大 commitment loss 权重；Usage < 50% → 启用码字重置 / 减小量化维度 |
| **Convergence** | 3k-10k | Recon MSE, Probe Acc | MSE 停滞 → 增加 FM 采样时间步 t 多样性；Probe Acc 低 → 检查 KV 提取 / 增加探针训练数据 |
| **Cycle Validation** | 10k+ | ODE Cycle Error | Error 高 → 增加 ODE 积分步数 / 检查 FM 参数化 Lipschitz 约束 |

### Step 4: 语义探针（"Is It Alive?" Test）

训练过程中（每 2k 步）冻结右脑，训练极简线性分类器：

- **输入**：FSQ 码字或中间 latent
- **任务**：预测对应 KV 来源 token 的实体类型（PER/LOC/ORG/NONE）或句法角色（SUBJ/OBJ/MOD）
- **意义**：如果 Recon MSE 很低但 Probe Acc 也很低，说明模型学到"像素级重建"但丢失语义。**这是最危险的假阳性。**

### Step 5: 结构探针（Phase 0 后期，mRNA 视角新增）

- **Attention Map 演化**：每 1k 步可视化右脑编码器 Attention Matrix，应从均匀/对角线模式演变为**块状+稀疏连接**（社区结构雏形）。
- **FSQ 码字互信息**：不同量化维度间互信息 $I(c_k; c_{k'})$ 应随训练**下降**（因子解耦）；同维度相邻 Token 互信息应**上升**（局部结构形成）。
- **单码字扰动**：改变 FSQ 序列一个码字，观察重构 KV 变化范围。局部变化 = 模块化结构形成；全局弥散 = 未稳定折叠。
- **码字交换**：互换两个语义相似实体的码字，检查重构 KV 是否保持功能等价（同义突变 = 结构鲁棒性）。
- **Recon-结构耦合**：Recon MSE 与 Spectral Gap 应同步改善。若 MSE 降而结构指标不动，说明模型用非结构化方式暴力记忆，检查双向注意力是否被短路。

---

## 三、 死亡线与逃生舱

Phase 0 必须有明确终止条件，避免"再调调就好了"的沉没成本陷阱。

### 3.1 死亡线（Kill Criteria）

满足任一条件，**立即停止当前配置，回退分析**：

1. 5k 步内 Velocity Loss 无任何下降趋势（排除 Warmup 期）。
2. FSQ Utilization 在 3k 步后仍 < 50%，且调整 commitment loss 无效。
3. Recon MSE > 3x Autoencoder 基准值，且持续 2k 步无改善。
4. ODE Cycle Error > 0.1，且增加积分步数无效（流场不可积）。
5. 频繁 NaN/Inf，且梯度裁剪、LR 调整、精度切换均无效。

### 3.2 逃生舱（Fallback Plan）

触及死亡线后按优先级尝试：

1. **降级 FM**：Flow Matching → 普通 VAE/AE，先验证 FSQ + 双向编码器。若 AE 成功 FM 失败 → 问题在流匹配。
2. **降级 FSQ**：FSQ → Gumbel-Softmax / Soft-VQ，先验证连续松弛。若 Soft 成功 Hard 失败 → 问题在量化。
3. **降级架构**：双向注意力 → 单向，验证是否方向性导致不稳定。
4. **终极保底**：放弃 FM+FSQ，回退 Gated GNN + RVQ。不够优雅但能保证产出。

---

## 四、 交付物清单

Phase 0 结束必须产出以下文档，而非仅一个 checkpoint：

1. **收敛报告**：Loss、MSE、Codebook Usage、Probe Acc 四条曲线的完整截图与分析。
2. **超参敏感性表**：至少 3 组 LR、2 组 FSQ 维度、2 组 ODE Solver 步数的结果对比。
3. **UMAP 演化图**：FSQ 码字空间从 Step 0 → 1k → 5k → 10k 的结构演变。
4. **Cycle Error 分布直方图**：证明 ODE 正反向一致性全局成立，而非仅某些样本。
5. **Go/No-Go 决策备忘录**：基于数据明确写出"进入 Phase 1"或"修改配置重跑 Phase 0"的结论及理由。

---

> **最后强调**：Phase 0 的数据不会骗人。如果 Loss 曲线丑陋、码本死寂、探针随机，**不要解释，不要合理化，不要相信"再训久一点就好了"**。
>
> SADKO 的"箭与环"之美，建立在右脑流形精确可逆的物理事实之上。Phase 0 就是验证这个物理事实。通过了，才有资格谈论图结构的生长；通不过，一切愿景皆为幻觉。
>
> 监控面板上的 Loss 曲线不是抽象数字——从分子视角看，它们是**转录速率、翻译效率、折叠自由能**。读懂它们，就读懂了 SADKO 的生命体征。
