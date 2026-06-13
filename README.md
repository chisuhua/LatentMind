# LatentMind — 原生多模态潜空间认知推理模型

> **定位**：物理 AI 时代的原生多模态认知推理芯片核心模型
> **架构**：基于 HRM-Text（潜空间递归推理）+ GRAM（多轨迹概率推理）
> **目标**：<1B 激活参数，端侧本地推理，原生视觉+文本+知识图谱联合推理

---

## 核心架构

```
图像 / 文本 / 知识图谱
  ↓
原生统一感知层（3层 CNN，~150M）
  ↓
分层递归潜空间引擎（~750M）—— HRM + GRAM
  ├── 高层（慢速）：全局逻辑、长程因果、宏观布局
  ├── 低层（快速）：局部高频特征、细粒度对齐
  └── 机器潜空间语言（内部 K 次循环，多轨迹采样）
  ↓
双流解码层（~100M）
  ├── 语言解释头（PrefixLM）
  └── 生成式语义头（轻量 DiT / Flow Matching）
  ↓
动作 / 语义图 / 自然语言解释
```

---

## 与 ChipForge APU 的关系

LatentMind 是 ChipForge APU 的**认知推理核**，运行在芯片上。

```
ChipForge APU
├── RISC-V 控制核
├── CUDA SIMT 计算核
├── Tensor Core 矩阵加速
└── LatentMind 认知核（软件栈最顶层）
```

---

## 项目结构

```
LatentMind/
├── src/ # 模型代码
│   ├── model/        # 核心模型（HRM+GRAM）
│   ├──感知层/        # 原生统一感知层（CNN）
│   ├──解码器/        # 双流解码器
│   └── utils/        # 工具
├── docs/             # 技术文档
│   ├── architecture.md    # 总体技术架构
│   ├── references/        # 外部研究参考笔记
│   │   ├── hrm-text.md   # HRM-Text 论文笔记（v1.0 backbone）
│   │   ├── gram.md       # GRAM 论文笔记（v1.5 扰动）
│   │   ├── trm.md        # TRM 笔记
│   │   ├── hrm-original.md  # HRM 原始 27M
│   │   ├── rrm-reward.md # RRM-A 奖励推理
│   │   ├── rrm-survey.md # RRM-B 谱系调查
│   │   ├── projects.md   # 相关项目元信息
│   │   ├── papers/       # 论文 PDF/HTML 源文件
│   │   └── README.md     # 索引
│   └── rfcs/             # 内部 R&D 提案
│       ├── README.md
│       ├── grr-300-gated-recursive-refiner.md  # 📝 草案
│       └── mr-300-micro-refiner.md              # 📝 草案
├── tests/            # 测试
└── examples/         # 示例
```

> 📦 区分 `docs/references/`（外部研究笔记）和 `docs/rfcs/`（内部 R&D 提案）。详见 [docs/rfcs/README.md](docs/rfcs/README.md)。

---

## 技术来源

- **HRM-Text**（Sapient，2026-05）：1.15B 参数，40B tokens训练，latent space 推理
- **GRAM**（Bengio + KAIST + Mila，2026）：生成式递归推理，多轨迹概率推理

---

## 开发状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 概念设计 | ✅ 完成 | 基于 HRM-Text + GRAM |
| 权重获取 | 🔨 进行中 | HRM-Text 已开源，需获取 |
| 感知层集成 | ⏳ 待开始 | 在 HRM backbone 上加原生感知 |
| v1.0 Demo | ⏳规划中 | 6 个月目标 |
| v1.5 完整方案 | ⏳ 规划中 | 12 个月目标 |

---

## 相关项目

| 项目 | 关系 |
|------|------|
| **minimind** | 文本认知推理（独立产品线）|
| **HydraForge** | 推理调度层（LatentMind 下游）|
| **ChipForge** | 芯片硬件（LatentMind 部署目标）|
| **AgenticLlama** | 推理引擎（Triton 优化）|