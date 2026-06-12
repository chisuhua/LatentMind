# LatentMind — Session Memory

## 项目信息
- **项目名**：LatentMind
- **创建日期**：2026-06-12
- **定位**：原生多模态潜空间认知推理模型
- **上级项目**：ChipForge APU（认知核）

---

## 技术来源

### HRM-Text（Sapient，2026-05-18）
- 1.15B 参数，40B tokens 训练，latent space 推理
- 训练成本：$1,500
- 已开源权重（需获取链接）
- 论文：https://sapient.inc/introducing-hrm-text

### GRAM（Bengio + KAIST + Mila，2026）
- OpenReview：https://openreview.net/forum?id=Vxu6kcIjwV
- PDF：https://openreview.net/pdf/cb0d33e74ecb64051d93d47865ebd10e7b5fafb6.pdf
- 核心：stochastic latent trajectories，多假设推理

---

## 架构决策

### v1.0（Demo）
- 直接复用 HRM-Text 预训练权重
- 只新增：原生感知层（3层 CNN）+ 语义图解码（轻量 DiT）
- 总参数：~1B

### v1.5（完整方案）
- 分层递归潜空间引擎（HRM + GRAM 双层叠加）
- 双流解码（语言 + 语义图 + 置信度）
- 多轨迹概率推理

---

## 待完成事项

- [ ] 获取 HRM-Text 预训练权重
- [ ] 评估 HRM-Text 在多模态场景上的基线效果
- [ ] 设计原生感知层（3层 CNN）
- [ ] 集成感知层到 HRM backbone
- [ ] 实现双流解码层

---

## 相关项目

| 项目 | 路径 | 关系 |
|------|------|------|
| minimind | `/workspace/project/minimind/` | 文本认知（独立产品线）|
| HydraForge | `/workspace/project/HydraForge/` | 推理调度层 |
| ChipForge | `/workspace/project/ChipForge/` | 芯片硬件 |
| AgenticLlama | `/workspace/project/AgenticLlama/` | Triton推理引擎 |

---

**最后更新**：2026-06-12