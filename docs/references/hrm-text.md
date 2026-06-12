# HRM-Text 参考资料

> **论文**：*HRM-Text: Efficient Pretraining Beyond Scaling*
> **作者**：Guan Wang*, Changling Liu*, Chenyu Wang, Cai Zhou, Yuhao Sun, Yifei Wu, Shuai Zhen, Luca Scimeca, Yasin Abbasi Yadkori†
> **机构**：Sapient Intelligence（新加坡） + MIT
> **arXiv**：2605.20613v1（2026-05-20）
> **官方发布**：2026-05-18

---

## 1. 一句话定位

**1B 参数的分层循环 LM，仅用 40B unique tokens + $1,500 预算，性能对标 2–7B 开源模型**。核心是"双时间尺度 H/L 循环 + MagicNorm + 任务完成目标 + PrefixLM"四件套，每个都对最终性能有显著贡献。

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| 官方博客 | https://sapient.inc/introducing-hrm-text |
| 产品页 | https://sapient.inc/hrm-text/ |
| arXiv | https://arxiv.org/abs/2605.20613 |
| arXiv HTML | https://arxiv.org/html/2605.20613v1 |
| HuggingFace | https://huggingface.co/sapientinc/HRM-Text-1B |
| GitHub | https://github.com/sapientinc/HRM-Text |
| HF Transformers 文档 | https://huggingface.co/docs/transformers/en/model_doc/hrm_text |
| HF Transformers PR | https://github.com/huggingface/transformers/pull/46025 |
| vLLM PR | https://github.com/vllm-project/vllm/pull/43098 |
| MLX-VLM PR | https://github.com/Blaizzy/mlx-vlm/pull/1238 |
| 数据准备工具 | https://github.com/sapientinc/data_io |

---

## 3. 核心创新

### 3.1 三大设计（论文 Table 3 消融已证实）

| 创新 | 作用 | 单独贡献（MMLU / MATH） |
|---|---|---|
| **HRM 双时间尺度循环** | 慢/快两栈替代标准 Transformer，深层"有效深度"远高于普通 Transformer | +6.9 / +8.7 |
| **任务完成目标** | loss 100% 在响应 token，prompt 不浪费容量 | +7.2 / +11.6 |
| **PrefixLM 注意力** | 指令端双向 + 响应端因果 → 单 forward 内等效 encoder-decoder | +5.4 / +1.3 |
| **三者叠加** | — | 60.73 / 56.16 |

### 3.2 HRM 循环结构

单次前向：2 个 H 周期 × 每个 H 周期 3 个 L 步骤 = 8 次栈迭代
- **z_H**：从 token embedding 初始化，慢速层（全局逻辑）
- **z_L**：固定初始化，快速层（局部特征）
- 最终 H 状态接线性 LM 头

```python
# HF Transformers 模型类 README 原文
z_H = embed(input_ids) * embedding_scale
z_L = z_L_init.expand_as(z_H)
for _ in range(H_cycles):     # 2
    for _ in range(L_cycles): # 3
        z_L = L_module(z_L + z_H)
    z_H = H_module(z_H + z_L)
return z_H
```

### 3.3 MagicNorm（前向 PostNorm + 反向 PreNorm）

利用 TBPTT 截断造成的前向/反向 horizon 不对称：

$$z_n = \text{Norm}\!\left(z_{n-1} + \sum_{l=1}^{L} \text{Sublayer}_l(\text{Norm}(\cdot))\right)$$

- 模块内部 PreNorm（保留 identity 路径，梯度流畅）
- 模块出口再 Norm（前向激活方差受 N 次约束，深层稳定）
- 反向时 BPTT 截断 → 梯度只穿过 K ≪ N 次出口 norm → 等效 PreNorm 优化稳定性

### 3.4 预热深度信用分配（Warmup BPTT）

- 原 HRM 固定 BPTT=1（只最后一步梯度）
- HRM-Text 改为 **K=2 → K=5 线性预热**：早期只 BPTT 最后 2 步，逐渐加到 5 步
- 灵感：生物时序学习 + 发展课程
- 附带好处：早期反向少 → 训练加速

---

## 4. 1B 模型架构参数

| 字段 | 值 |
|---|---|
| 总参数 | **~1.15B**（BF16 文件 2.36 GB） |
| 每 H/L 栈层数 | **16** |
| 隐藏维度 | **1536** |
| 注意力头 | 12（MHA, head_dim 128） |
| MLP 中间层 | 4096 |
| 上下文 | **4096** |
| 词表 | **65,536**（BPE） |
| 位置编码 | RoPE, θ=10000 |
| 激活 | SwiGLU |
| 归一化 | MagicNorm + 无 γ RMSNorm |
| 注意力门控 | Sigmoid 输出门（Qwen3Next 风格） |
| 精度 | bfloat16 |
| 初始化 | LeCun normal（embedding × 1/σ） |
| 嵌入缩放 | `embedding_scale = 1 / initializer_range` |

> **架构预设**（arch/size 配置文件）：
> B=12L/1024H/8A，L=24L/1280H/10A，**XL=32L/1536H/12A（默认 1B）**，XXL=72L/1792H/14A，XXL_wide=32L/2560H/20A
>
> HF 端 `num_hidden_layers` 被 inflate 为 `16 × H_cycles × (L_cycles+1)` —— 权重共享，仅在 cycle slot 维度上展开

---

## 5. 训练配方

| 项 | 值 |
|---|---|
| 优化器 | **Adam-atan2**（β₁=0.9, β₂=0.95, wd=0.1） |
| 学习率 | 2.2×10⁻⁴，**2000 步预热 + 之后保持常数（无衰减）** |
| 梯度裁剪 | **不应用** |
| EMA | 0.9999（发布权重 = EMA checkpoint） |
| 全局 batch | 196,608 tokens |
| 序列长度 | 4096 |
| L_bp_steps（BPTT） | 训练时 K=2 → K=5；推理可左 pad 到 L_cycles |
| 硬件 | **2 节点 × 8 H100**，**46 小时** |
| 成本 | $1,472（按 $2/H100·小时） |
| 训练总 token | 60B（40B unique × 4 epochs） |
| 并行化 | PyTorch FSDP2 |
| 稳定性保障 | **无**（不切 spike、不中断恢复） |

---

## 6. 数据集组成

**初始语料**：~176.5B tokens / 593.7M 文档（开源公共数据）
**训练**：40B unique tokens × 4 epochs = 60B tokens，**SeqIO 风格分层采样**

| 类型 | 数据集 | Tokens | 训练条件 |
|---|---|---|---|
| 通用指令 | FLAN, Tasksource, NoRobots | 138.7B | direct / cot |
| 维基重写 | SYNTH | 21.7B | synth, direct/cot |
| 数学/推理 | Platypus, Principia, OpenMathInstruct2, NuminaMath, OmniMATH | 6.8B | synth / synth, cot |
| 符号 | DM Math, AMPS, Sudoku-Extreme | 6.2B | direct |
| 推理 | AceReason, OpenThoughts2 | 2.4B | synth, cot |
| 教科书 | TextbookReasoning | 358.1M | synth, cot |
| 网页指令 | NaturalReasoning, WebInstruct-verified, AMPS-khan | 375.1M | noisy / direct / cot |

**关键设计**：
- **4 个条件标签**：`direct`、`cot`、`synth`、`noisy`，前置于指令用于条件化输出风格
- **去除 `<think>...</think>` 标签**：刻意剥离 RLVR 显式长 CoT 痕迹
- **响应端 NLL**：仅在响应 token 上计算 loss

---

## 7. 基准结果（论文 Table 4）

| 模型 | 架构 | FLOPs(10²¹) | Tokens(T) | MMLU | ARC-C | DROP | GSM8K | MATH |
|---|---|---|---|---|---|---|---|---|
| **HRM-Text 1B** | Recurrent | **1** | **0.06** | **60.7** | **81.9** | **82.2** | **84.5** | **56.2** |
| Huginn 3.5B | Recurrent | 127 | 0.8 | 31.4 | 38.2 | 17.8 | 34.6 | 12.6 |
| Olmo3 7B | Dense | 252 | 6 | 65.8 | 81.6 | 71.5 | 75.5 | 40.0 |
| Llama3.2 3B | Dense | 162 | 9 | 58.0 | 69.1 | 45.2 | 77.7 | 48.0 |
| Gemma3 4B | Dense | 96 | 4 | 59.6 | 56.2 | 60.1 | 38.4 | 24.2 |
| Qwen3.5 2B | Dense | 432 | 36 | 64.5 | 81.0 | 30.8 | 53.0 | 34.2 |
| Ouro 1.4B | Recurrent | 259 | 7 | 67.4 | 60.9 | 49.7 | 78.9 | 22.4 |

→ 在少 100–900× tokens、少 96–432× FLOPs 的预算下，HRM-Text 1B 与 2–7B 开源模型有竞争力。

**数据污染**：1B 模型仅 DROP 在 n=13 上有边际污染（n=20 不显著）；干净子集（0% 污染）DROP 仍达 81.1。

---

## 8. 开源权重

| 项 | 值 |
|---|---|
| HuggingFace 仓库 | https://huggingface.co/sapientinc/HRM-Text-1B |
| **许可证** | **Apache 2.0** |
| 创建 | 2026-05-17 |
| 平台标签 | pre-alignment, non-chat, non-instruction-tuned, prefix-lm, hrm, en |
| dtype | BF16 单文件 2.36 GB |
| 特殊 token | `pad=<|endoftext|>`, `eos=<|box_end|>` |
| Chat template | **无**（base model） |
| 变体 | **仅 1B base**（0.6B 内部模型未单独发布） |

**消费端支持矩阵**：

| 平台 | 状态 | 备注 |
|---|---|---|
| HF Transformers ≥ 5.9.0 | ✅ 原生 `hrm_text` 类 | PR #46025 已合并 |
| vLLM | ⏳ PR #43098 review | 自动检测 non-causal → 禁用 chunked prefill + prefix caching |
| MLX-VLM | ✅ PR #1238 已合并 | 仅文本路径 |

---

## 9. GitHub 仓库结构

URL：https://github.com/sapientinc/HRM-Text（~1K stars，2026-05-18）

| 文件 | 作用 | 对 LatentMind 价值 |
|---|---|---|
| `pretrain.py` | FSDP2 训练主循环 | **高** — 复用训练框架 |
| `models/baselines/hrm_nocarry_bp_warmup.py` | **主架构** | **高** — backbone |
| `models/layers.py` | RoPE、gated MHA、SwiGLU、静态 KV cache | **高** |
| `models/lm_head.py` | scaled embedding、output head、CE loss | 中 |
| `models/flash_attention_prefixlm_v2.py` | 两段式 PrefixLM 注意力 | **高** — 多模态融合 |
| `dataset_new.py` | PrefixLM packed dataset | 中 |
| `multipack_sampler.py` | 分布式 multipack 采样器（LPT 分配） | 中 |
| `simple_inference_engine.py` | 简单推理引擎（连续批处理） | **高** — 推理起点 |
| `conversion/convert_to_hf.py` | FSDP2 → HF 格式导出 | 中 |
| `evaluation/` | 各 benchmark wrapper | 中 |
| `config/` | Hydra 配置 | 高 |
| `docker/Dockerfile` | 锁定 CUDA / PyTorch / FA3 | 中 |
| `scripts/prepare_sft_data.py` | SFT 数据准备 | 中 |

---

## 10. 复现风险（针对 LatentMind 多模态场景）

### 10.1 文本-only 假设

| 假设 | 多模态影响 | 必须重做 |
|---|---|---|
| 输入是 1D token id 序列 | 图像/视频是 2D 空间 | **感知层**（3 层 CNN） |
| 词表 BPE 65,536 | 视觉 token 无 BPE | 词表扩展 / hybrid embedding |
| PrefixLM 掩码 0/1 区分指令/响应 | 图像块是 prefix 还是 response？跨模态注意力？ | **mask 构造** |
| LM 头只输出 token logits | 语义图需要 diffusion / flow matching | **双流解码头** |
| 4 个文本 condition 标签 | 需要"图像条件"等额外 token | 扩展 condition 系统 |
| RoPE θ=10000 / 4K context | 视觉 token 可能要 8K+ | 扩位置编码或上插 |
| `token_type_ids` 1D 同长 | 图像块独立 mask 模式 | 扩 PrefixLM attention helper |
| **无 instruction tuning** | 文本就是 base | 多模态需 SFT 阶段 |
| **单语（仅英语）** | — | 多模态需决定是否多语 |
| **无代码训练** | 代码任务"low single digits" | 如需代码能力须 SFT |

### 10.2 推理侧硬性约束

1. **FlashAttention 不支持 PrefixLM**：必须 SDPA 或 flex_attention
2. **vLLM 禁用 chunked prefill + prefix caching**（自动）
3. **vLLM Pipeline parallelism + LoRA 暂不支持**
4. **多轮对话**：用户段双向 + 助手段因果的混合 attention pattern

### 10.3 训练侧脆弱性

- 无 gradient clipping → spike 不切 → 监控 + checkpoint 必做
- 单次连续运行（不恢复）→ 多模态训练时长可能 7–10 天，建议增加 checkpoint
- 学习率保持常数（无衰减）→ 多模态数据少时可能需更激进衰减

### 10.4 不在论文保证范围内的特性

- ❌ 多模态融合（**完全没有**）
- ❌ 图像/视频输入
- ❌ 结构化输出（语义图、JSON schema）
- ❌ 长上下文（>4K 未验证）
- ❌ 流式 + 工具调用
- ❌ Function calling / Agentic
- ❌ 安全对齐（pre-alignment）
- ❌ >1B 规模验证（作者自承）

---

## 11. 复用建议

| 类别 | 内容 |
|---|---|
| **直接复用** | backbone（`hrm_nocarry_bp_warmup.py`）、`layers.py`、`flash_attention_prefixlm_v2.py`（扩展为视觉前缀）、`pretrain.py`、`simple_inference_engine.py` |
| **必须修改** | 1) 插入 3 层 CNN 感知层；2) 扩展词表或维护独立视觉 embedding；3) 双流解码头（语言 PrefixLM + 语义图 DiT/Flow）；4) mask 构造支持"图像块=prefix，文本=prefix，响应=causal" |
| **集成 GRAM 时** | H 模块 residual 后加可学习 μ_θ, σ_θ；注入 ε ~ N(μ, σ²I)；变分 ELBO 训练；**注意 GRAM 论文无 1B 验证**，需先小规模消融 |
| **量化（针对 ChipForge APU）** | backbone FP16（不用 Q4）；感知层/动作头 INT8/Q4；目标 <1 GiB |
| **必做验证实验** | 感知层→backbone 对齐损失；PrefixLM 在图像前缀上的等效性；GRAM 多轨迹 1B 可扩展性；无 RAG 时 1B 事实知识容量 |

---

## 12. 相关工作一览

| 名称 | 与 HRM-Text 关系 | 链接 |
|---|---|---|
| **GRAM** | LatentMind 第二个核心依赖。变分随机化扩展 HRM 风格循环，**实验只跑到 10–11M**（Sudoku/N-Queens），**1B 未验证** | [arXiv:2605.19376](https://arxiv.org/abs/2605.19376) / [项目页](https://ahn-ml.github.io/gram-website) |
| HRM（原始） | 2025-06 27M 符号推理版（Sudoku/maze/ARC），Sapient 姊妹仓库 | https://github.com/sapientinc/HRM |
| Huginn 3.5B | latent recurrent LM（HRM-Text 对比基线） | — |
| Ouro 1.4B | looped LM（HRM-Text 对比基线） | — |
| TRM | 共享 H-L 参数的小型递归模型（7M ARC-AGI 45%），HRM-Text 1B 时训练不稳定 | arXiv:2510.00471 |
| HRM-MoE | 社区 MoE 扩展（64×8 路由，~5.4B 总/~1.18B 激活） | https://github.com/XiaoYee/HRM-MoE |
| Universal Transformer | 早期循环深度 Transformer | — |

---

## 13. 关键引用块（写论文/汇报时直接抄）

> 论文摘要原文：
> "We present HRM-Text, a 1B-parameter hierarchical recurrent model trained from scratch using only 40B unique tokens and a budget of $1,500, achieving 60.7% MMLU, 81.9% ARC-C, 82.2% DROP, 84.5% GSM8K, and 56.2% MATH — competitive with 2–7B open-source models despite using 100–900× less training data and 96–432× less compute."

> MagicNorm 原文：
> "MagicNorm, which exploits the asymmetry between the forward and backward computational horizons induced by truncated backpropagation through time (TBPTT)."

> 性能核心论证：
> "At 1B parameters, HRM-Text is trained on 40B unique tokens (60B total with 4 epochs) and reaches MMLU 60.7 — 5–8 points above similarly sized open baselines, with a training cost of approximately $1,500."

---

**最后更新**：2026-06-13
**信息源**：arXiv 2605.20613v1、HF 模型卡、官方博客、GitHub README、HF Transformers PR、第三方分析
