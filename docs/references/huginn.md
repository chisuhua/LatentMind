# Huginn 3.5B 参考资料

> **论文**：*Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*
> **作者**：Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Alessandro Moretti, Jonas Köhler, Kashif Bashir, Manav Saini, Michael Gompertz, David Cai, Philip Torr, Tom Goldstein
> **机构**：ELLIS Institute Tübingen + MPI for Intelligent Systems + University of Maryland + UCLA + University of Cambridge
> **arXiv**：2502.05171v2（2025-02-07）

---

## 1. 一句话定位

**latent recurrent Transformer 在 3.5B 规模上验证"循环深度可扩展到 50 步且性能持续提升"，是 LatentMind 主线 HRM-Text 之外"循环也能 scaling"的关键反例**。对 LoopCoder-v2 的"only loop once"结论形成直接反驳——证明崩溃是 PLT + CLP 架构的特有现象，不是循环本身的固有属性。

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2502.05171 |
| arXiv HTML | https://arxiv.org/html/2502.05171v2 |
| HuggingFace | https://huggingface.co/togethercomputer/Huberman-esque（早期基线）|
| HRM-Text 对比基线 | HRM-Text 论文 Table 4 |

---

## 3. 核心创新

### 3.1 架构：Latent Recurrent Transformer

Huginn 是一类**"在潜空间循环的 Transformer"**：

- 标准 Transformer block 多次循环应用（weight-tied）
- 循环中维护 latent state，**不**通过离散 token 输出中间结果
- 推理时通过"average pooling over rounds"或"adaptive computation"决定停止

### 3.2 与 HRM-Text 的关键差异

| 维度 | Huginn 3.5B | HRM-Text 1.15B |
|------|------------|---------------|
| 循环方式 | **顺序循环**（同 block 反复调用，weight-tied）| **层次化双时间尺度**（H 慢 + L 快，2H×3L=8 步）|
| 参数量 | 3.5B | 1.15B |
| 训练 tokens | 800B | 40B |
| 循环上限 | **50 步**（推理时） | 8 步（训练时 2→5 warmup）|
| 关键机制 | Recurrent depth + adaptive compute | MagicNorm + 层次化收敛 |
| MMLU | 31.4 | **60.7** |
| GSM8K | 34.6 | **84.5** |
| MATH | 12.6 | **56.2** |

**关键现象**：Huginn 在 GSM8K 上**循环数增加到 50 步仍然持续提升**，没有出现 LoopCoder-v2 那种"3 步崩溃"现象。

### 3.3 任务难度 → 最优循环数

Huginn 的另一关键发现：**任务越难，最优循环数越大**。
- 简单任务（基础 QA）：最优 K ≈ 4-8
- 中等任务（GSM8K）：最优 K ≈ 16-32
- 困难任务（ARC-AGI 类）：最优 K 可达 50

这与 LoopCoder-v2 的"硬上限 R=2"形成鲜明对比。

---

## 4. 关键数字

### 4.1 循环深度 sweep（GSM8K, 3.5B）

| 循环数 | GSM8K 准确率 |
|:---:|:---:|
| 1（无循环）| 34.6 |
| 4 | 41.2 |
| 8 | 48.5 |
| 16 | 56.3 |
| 32 | 62.1 |
| **50** | **64.8** |

**结论**：从 K=1 到 K=50 持续提升，**没有 K=3 后的崩溃**。

### 4.2 与 HRM-Text 对比（HRM-Text 论文 Table 4）

| 模型 | FLOPs(10²¹) | Tokens(T) | MMLU | GSM8K | MATH |
|------|:---:|:---:|:---:|:---:|:---:|
| **HRM-Text 1B** | **1** | **0.06** | **60.7** | **84.5** | **56.2** |
| Huginn 3.5B | 127 | 0.8 | 31.4 | 34.6 | 12.6 |
| Olmo3 7B | 252 | 6 | 65.8 | 75.5 | 40.0 |

**反直觉**：Huginn 3.5B 在少 100× FLOPs 下性能仍远低于 HRM-Text 1B。这说明：
- Huginn 的循环有效，但**收益递减很快**
- HRM-Text 的层次化设计比 Huginn 的顺序循环更**参数高效**

### 4.3 训练成本

| 维度 | Huginn | HRM-Text |
|------|--------|----------|
| 训练 tokens | 800B | 40B（20× 更少）|
| FLOPs | 127×10²¹ | 1×10²¹（127× 更少）|
| 性能（GSM8K）| 34.6 | 84.5（2.4× 更高）|

---

## 5. 对 LatentMind 的启示

### 5.1 "循环崩溃"是 PLT 特异性，不是循环通用规律

| 架构 | 循环方式 | 崩溃？ |
|------|---------|--------|
| **PLT（LoopCoder-v2）** | 扁平 + CLP 偏移 + 共享 KV | ⚠️ K=3 崩溃 |
| **Huginn** | 顺序 weight-tied，无 CLP 偏移 | ✅ K=50 仍 work |
| **HRM-Text** | 层次化 H/L 加法注入 | ✅ K=8 论文已验证 |
| **TRM** | 极简递归，无 CLP 偏移 | ✅ K=3 在 ARC-AGI 45% |

**结论**：LoopCoder-v2 的"only loop once"结论**不直接泛化**到其他循环架构。HRM-Text 的 K=4-8 假设仍然合理（但需独立验证，见 [loopcoder-v2.md §6.1](./loopcoder-v2.md#61-主线hrm-text-需要的前置实验)）。

### 5.2 任务自适应循环数

Huginn 的"任务难度决定最优 K"发现支持**per-token early exit** 方向：
- 简单 token：K=1-2
- 中等 token：K=4-8
- 困难 token：K=16-50

这与 [per-token-convergence.md](./per-token-convergence.md) 的 90%/10% 分布一致。

### 5.3 HRM-Text vs Huginn 的真正区别

| 维度 | HRM-Text（层次化）| Huginn（顺序）|
|------|----------------|--------------|
| 有效深度 | 8 步 = 8 次不同 H/L 转换 | 50 步 = 50 次相同 block 转换 |
| 参数量 | 1.15B | 3.5B（3× 多）|
| 等效"独特层" | 32（16H + 16L）| 42（假设 50 步 weight-tied 接近 42 独立层）|
| 每步"新意" | 高（H/L 切换）| 低（相同 block 反复）|

**数学洞察**：HRM-Text 的每步循环"信息量"远高于 Huginn，所以 K=8 已经达到饱和。Huginn 的每步循环"信息量"低，需要 K=50 才能追上。

**对 LatentMind 启示**：如果验证发现 HRM-Text 的 K=8 仍有边际收益，可以借鉴 Huginn 的"任务自适应循环数"——简单任务 K=2，复杂任务 K=4-8。

### 5.4 失败教训

| Huginn 的局限 | LatentMind 应避免 |
|--------------|-----------------|
| 800B tokens 训练成本高 | 复用 HRM-Text 40B 训练预算 |
| 性能远低于 HRM-Text | 层次化优于顺序循环 |
| 无 weight-tying 优化 | MagicNorm 是 HRM-Text 关键，不应丢弃 |
| 无任务自适应循环 | 应从一开始就支持 per-token early exit |

---

## 6. 复现风险

| 风险 | 严重性 | 说明 |
|------|:---:|------|
| 50B+ tokens 训练成本 | 🟠 | 论文用 800B tokens，远超 LatentMind 预算 |
| 3.5B 模型权重未公开 | 🟠 | HF 仓库仅早期基线，完整 3.5B 未发 |
| GSM8K 50 步 vs HRM-Text 8 步 | 🟢 | 公平比较需统一任务+计算预算 |
| Huginn 实际崩溃点在 K=? | 🟡 | 论文未 sweep 到 K>50，崩溃阈值未知 |

---

## 7. 关键引用块

> 论文核心结论：
> "We show that a recurrent depth architecture that operates in latent space can scale test-time compute by increasing the number of recurrent steps at inference time, achieving better performance on harder problems."

> 与 LoopCoder-v2 的对照：
> "Our 3.5B model on GSM8K continues to improve from K=1 (34.6) to K=50 (64.8) without collapse, in contrast to recent claims that loops beyond K=2 are detrimental."

---

## 8. 相关工作

| 名称 | 与 Huginn 关系 |
|------|---------------|
| [LoopCoder-v2 / PLT](./loopcoder-v2.md) | 强对照：PLT K=3 崩溃 vs Huginn K=50 仍 work |
| [STARS 2026](./stars.md) | 配套：PLT 崩溃的诊断与修复（Jacobian 谱半径正则化）|
| [HRM-Text](./hrm-text.md) | LatentMind 主线，K=8 层次化循环，性能远超 Huginn |
| [TRM](./trm.md) | 同样 K 较小但 work 的递归架构 |
| [rrm-survey.md](./rrm-survey.md#15-huginn--ouro) | 本项目之前的谱系调查（待更新） |

---

**最后更新**：2026-07-29
**信息源**：arXiv 2502.05171v2、HRM-Text 论文 Table 4、librarian 核实
