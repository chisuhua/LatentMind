# LoopCoder-v2 / PLT 参考资料

> **论文**：*LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling*
> **作者**：Jian Yang (北航), Shawn Guo, Wei Zhang, Tianyu Zheng, Yaxin Du, Haau-Sing Li, et al.
> **机构**：北京航空航天大学 + IQuest Research + 朗舟科技 + 中国人民大学
> **arXiv**：2606.18023v1（2026-06-16）
> **代码**：[github.com/CSJianYang/LoopCoder](https://github.com/CSJianYang/LoopCoder)
> **权重**：[huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2)

---

## 1. 一句话定位

**提出 PLT（Parallel Loop Transformer），证明"循环次数并非越多越好"——7B 模型在 SWE-bench Verified 上"只多循环 1 次"获得 50% 提升，但循环 3 次反而崩溃到基线以下。CLP + G-SWA 让循环变成零延迟旋钮**。这是首次对"循环越多越深"假设的大规模实证反驳。

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2606.18023 |
| arXiv HTML | https://arxiv.org/html/2606.18023v1 |
| GitHub | https://github.com/CSJianYang/LoopCoder |
| HuggingFace | https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2 |
| 原始 PLT 论文 | arXiv:2510.24824（PLT 架构首发） |

---

## 3. 核心创新

### 3.1 PLT 架构（Parallel Loop Transformer）

解决传统循环 Transformer 的两个工程难题：

| 难题 | PLT 解法 | 效果 |
|------|---------|------|
| 串行依赖导致延迟高 | **CLP（Cross-Loop Position Shift）**：跨循环位置偏移打破串行依赖 | 多次循环可**并行计算**，延迟接近单次推理 |
| KV 缓存随循环次数线性增长 | **G-SWA（Gated Sliding Window Attention with Shared KV）**：共享首循环 KV | 显存占用**几乎不随循环数增长** |

### 3.2 CLP（Cross-Loop Position Shift）

**核心公式**（论文 Eq. 2）：

$$B^{(r)} = \text{Embed}(x) + \text{shift}(\mathbf{h}^{(r-1)}), \quad \mathbf{h}^{(r)} = f_\theta(B^{(r)})$$

其中 `shift` 是右移一位：`shift(h^(r-1))_i = h^(r-1)_{i-1}`，且 `h^(r-1)_0 = 0`。

**关键**：第 r 轮循环中，token i 的输入是"自己的 embedding + 上一轮 token i-1 的 hidden state"，**而不是**自己的上一轮状态。这打破了"同一位置跨轮串联依赖"，使得 token i 的第 r 轮可以和 token i-1 的第 r+1 轮在同一前向中并行计算。

**代价**：每个 loop 边界引入固定位置偏移。论文用 $\Omega^{(r)}$ 量化：

$$\Omega^{(r)} = \frac{1}{S}\sum_i \|\mathbf{h}^{(r-1)}_i - \mathbf{h}^{(r-1)}_{i-1}\|_2$$

$\Omega^{(r)}$ 跨 loop 几乎恒定，意味着**每轮循环都支付同样大小的"位置偏移税"**。

### 3.3 G-SWA（Gated Sliding Window Attention with Shared KV）

**核心公式**（论文 Eq. 1）：

$$\tilde{y}^{(r)} = g \odot y_{\text{global}}^{(r)} + (1-g) \odot y_{\text{local}}^{(r)}, \quad g = \sigma(f_{\text{gate}}(\text{RMSNorm}(\mathbf{h})))$$

- $y_{\text{global}}^{(r)}$：对**冻结**的 $K_{\text{share}}, V_{\text{share}}$（来自 loop 1 的 KV cache）做 full-context attention
- $y_{\text{local}}^{(r)}$：对当前 loop 的 KV 做 sliding-window attention（窗口大小 w=64）
- $f_{\text{gate}}$：每个 attention head 一个标量，head-wise linear layer，输入经 RMSNorm

**KV cache 占用**：首次循环的 KV cache 冻结共享给所有后续循环。后续循环的 sliding-window 只缓存最近 w 个 token 的 KV entry。总 KV cache = $O(L \cdot S \cdot d)$ 与 loops 数 R 无关（对比普通 loop transformer 的 $O(R \cdot L \cdot S \cdot d)$）。

### 3.4 架构参数

| 参数 | 值 |
|------|------|
| 参数量 | ~7B |
| Hidden size | 5120 |
| Shared layers | 14 |
| Attention heads | 40 |
| KV heads | 8 |
| Head dim | 128 |
| Intermediate size | 27648 |
| Activation | SwiGLU |
| Norm | RMSNorm |
| RoPE theta | 500000 |
| Vocab | 76800 |
| Max seq len | 131072 |

---

## 4. 关键数字

### 4.1 核心 Loop-Count Sweep（SWE-bench Verified, 7B）

| 模型 | R=1（基线）| R=2 | R=3 | R=4 |
|------|:---:|:---:|:---:|:---:|
| LoopCoder-v2 | 43.0 | **64.4** | 27.6 | 22.4 |

**非单调性极其显著**：R=2 提升 21.4 分，R=3 跌到基线以下，R=4 更差。

### 4.2 跨 Benchmark 一致性

| Benchmark | R=1 | R=2 | R=3 | R=4 |
|-----------|:---:|:---:|:---:|:---:|
| HumanEval+ | 81.1 | 84.1 | 75.0 | — |
| MBPP+ | 69.5 | 73.9 | 69.8 | — |
| MultiPL-E | 38.0 | 46.1 | 43.3 | — |
| LiveCodeBench | 27.4 | 35.4 | 28.6 | 24.5 |
| SWE-bench Verified | 43.0 | 64.4 | 27.6 | 22.4 |
| Multi-SWE | 14.0 | 31.0 | 11.0 | 9.3 |
| SWE-bench-CC | 26.3 | 34.2 | — | — |
| Terminal-Bench | 11.2 | 21.0 | — | — |
| BFCL | 35.3 | 34.5 | — | — |
| API-Bank | 35.3 | 40.1 | — | — |

**结论**：R=2 在**所有** benchmark 上均优于 R=1，R=3 在**所有** benchmark 上均退化。**模式不是单 benchmark 噪声**。

### 4.3 三个崩溃机制（论文诊断）

| 机制 | 描述 | 启示 |
|------|------|------|
| **表示空间坍缩** | Effective rank 在 loop 2 达峰，之后下降 | 后续循环在低维空间操作，新计算能力枯竭 |
| **振荡而非收敛** | $\cos\theta^{(r)} < 0$ 在 loop 2 后 | 连续循环方向相反，不是精细调整而是来回振荡 |
| **固定位置偏移税** | $\Omega^{(r)}$ 跨 loop 几乎恒定 | 每轮"税"固定，但收益递减 → 净效果变负 |

**关键澄清**：这不是"训练分布不匹配"（因为训练和推理的 loop 数一致），也不是"梯度爆炸/消失"（用了梯度裁剪和 bf16）。**这是 PLT 架构固有的增益-成本权衡**。

### 4.4 与"多一层微调"的区别

Loop 是**共享参数**的重复应用，fine-tuning 是**改变参数**。PLT 的 loop 并不增加参数量，也不改变权重。Schwethelm 2026 的 scaling law 给出 $\varphi = 0.46$：**1 次 loop 约等于 0.46 个 unique layer 的容量增益**（Hyperconnections 可提升到 $\varphi = 0.65$）。

---

## 5. 与 HRM-Text 对比

### 5.1 架构差异

| 维度 | LoopCoder-v2 (PLT) | HRM-Text |
|------|-------------------|----------|
| Loop 结构 | 扁平 R 次（R=1,2,3,4）| 层次化 2 H-cycles × 3 L-cycles = 8 步 |
| 参数共享 | 14 层共享 block | 两个独立的 H/L 模块，各有 16 层 |
| 参数量 | 7B | 1.15B |
| 训练 tokens | 18T | 40B |
| Loop 间信息流 | CLP 偏移（右移一位）+ G-SWA 门控 | 加法注入 $z_L + z_H$ |
| KV 缓存 | 共享首次循环 KV + sliding window | 标准动态缓存 |
| 梯度策略 | 标准 BPTT | 截断 BPTT，warmup 从 2→5 步 |
| 关键创新 | 并行化推理，接近单次延迟 | MagicNorm + 层次化收敛 |

### 5.2 威胁评估

**这不是直接威胁**：
1. **架构不同**：PLT 的扁平 loop + CLP 偏移引入的"位置税"是 HRM-Text 没有的机制。HRM-Text 使用的是加法注入 $z_L + z_H$，没有偏移。
2. **规模不同**：PLT 是 7B / 18T tokens，HRM-Text 是 1B / 40B tokens。
3. **任务不同**：PLT 专注代码生成/SWE，HRM-Text 专注 reasoning/MMLU/GSM8K/MATH。
4. **无直接比较**：两篇论文未互相引用，无共同基准对比。

**但这是重要警告**：
1. **TRM 分析**（2026）已发现 TRM 的"大部分性能在第一次递归步骤就达到"——这对 HRM-Text 的 8 步递归价值提出质疑
2. **Per-token convergence** 证据显示大部分 token 在 6 轮内收敛——HRM-Text 的 8 步可能接近上限
3. **Scaling law** 证据（$\varphi=0.46$）表明循环的边际收益远低于完整参数

### 5.3 关键开放问题

| 问题 | 当前状态 | 需要验证 |
|------|---------|---------|
| "Only Loop Once" 在 70B 模型上是否成立？ | ❌ 论文未测试 | 必须 sweep 1B/7B/13B/70B |
| 3 圈崩坍的**数学原因**？ | 论文给出 3 个诊断 | 需要独立复现确认 |
| PLT 的 G-SWA 共享 KV 是否损失了"循环带来的深度增益"？ | 未量化 | 消融：无 G-SWA vs 有 G-SWA |
| 与 CoT 步数的关系？循环是否替代了 CoT 步数？ | 未讨论 | CoT 长度 vs Loop 数 |

---

## 6. 对 LatentMind 的启示

### 6.1 主线（HRM-Text）需要的前置实验

| 实验 | 目标 | 周期 |
|------|------|------|
| **K-sweep on HRM-Text** | K ∈ {2, 4, 6, 8, 12} 跑 ARC-AGI / GSM8K / MATH | 1-2 周 |
| **层次 vs 扁平对比** | 2H×3L=8 vs flat 8，验证层次结构是否免疫崩溃 | 1 周 |
| **Per-token 收敛分析** | 监控每个 token 多少步收敛 | 1 周 |
| **STARS 改造** | 移植谱半径正则化 | 1 周 |

**关键决策点**：如果 HRM-Text 在 K=4 出现退化，**不要改 K，改 LayerNorm 位置或加谱正则化**。

### 6.2 共享感知主干的融合点

PLT 的 CLP 偏移+共享 KV 思路**可以移植到 HRM-Text 的 H/L 循环**——把 K=8 的串行循环改成 K=8 的并行循环，端侧推理延迟从 8x 降为 1x。这是**端侧可行性的关键**。

### 6.3 反向利用"只多一次循环"

LoopCoder-v2 的核心发现可以**反过来用**——不是"循环越多越好"，而是"**多一次自检**就足够"。LatentMind 主线可以设计为：
- K=2 推理（首轮生成 + 一次校验）
- 校验通过 → 输出；校验不通过 → 触发 GRAM 多轨迹（见 [gram.md](./gram.md)）

这恰好契合 GRAM 的"多轨迹逃逸"机制（见 [gram.md](./gram.md)）。

### 6.4 SADKO 借鉴

SADKO 的 Hippo MemPool（见 [../research/sadko-v2-architecture.md](../research/sadko-v2-architecture.md)）可以借鉴 PLT 的 **G-SWA 共享 KV 机制**——Memory KV 池在"压缩-读取"循环中共享，避免 KV 缓存随循环次数增长。

---

## 7. 复现风险

| 风险 | 严重性 | 说明 |
|------|:---:|------|
| 仅 7B 验证 | 🟠 | 1B/13B/70B 未测，规模 sweep 缺失 |
| 无标准差报告 | 🟠 | 64.4 vs 27.6 差异巨大，但论文未报多次运行 std |
| 仅 PLT 架构测试 | 🟠 | 结论不直接泛化到 HRM-Text / Huginn / TRM 等 |
| 未在 ARC-AGI 评估 | 🟡 | 循环架构的关键基准，缺失 |
| 无消融：移除 CLP 后的循环是否还崩？ | 🟠 | 关键对照实验缺失 |

---

## 8. 关键引用块

> 论文标题与核心结论：
> "LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling"

> 7B 关键数字：
> "On SWE-bench Verified, our 7B baseline achieves 43.0%. With only one extra loop, performance jumps to 64.4% (+50%). However, when loops increase to 3, performance collapses to 27.6%, even worse than the no-loop baseline."

> PLT 设计动机：
> "Traditional looped transformers face two engineering challenges: serial dependency leads to high latency, and KV cache grows linearly with loop count. PLT solves these via CLP and G-SWA, making loop count a low-cost knob."

---

## 9. 相关工作

| 名称 | 与 LoopCoder-v2 关系 |
|------|---------------------|
| [Huginn 3.5B](./huginn.md) | 反例：3.5B 模型无 CLP 循环可到 50 步无崩溃 |
| [STARS 2026](./stars.md) | 配套：循环崩溃的诊断（LayerNorm 位置）+ 修复（Jacobian 谱半径正则化）|
| [Per-Token Fixed-Point](./per-token-convergence.md) | 配套：90% token 6 步收敛，10% 需要 8 步——动态 K 证据 |
| [HRM-Text](./hrm-text.md) | LatentMind 主线 backbone，K=8 层次化循环 |
| [TRM](./trm.md) | 反例：极简递归，K=3 在 ARC-AGI 上 45% |
| [Hyperloop Transformer](https://arxiv.org/abs/2604.21254) | 跨层超连接，$\varphi$ 可从 0.46 提升到 0.65 |

---

**最后更新**：2026-07-29
**信息源**：arXiv 2606.18023v1、GitHub 仓库 README、HF 模型卡、librarian 详细核实
