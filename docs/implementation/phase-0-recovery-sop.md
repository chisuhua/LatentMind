# Phase 0 失败回退 SOP 与决策树

> **一句话定位**：Phase 0 失败时的**标准操作程序（SOP）**——包含失败分类、修复路径、回退方案、紧急升级流程，配合 [phase-0-implementation-guide.md](./phase-0-implementation-guide.md) 使用。
> **性质**：协调文档的**配套应急手册**
> **最后更新**：2026-07-29（v1.0 新增）

---

## 0. 文档定位

| 文档 | 关系 |
|------|------|
| [logos-sadko-64m-coordination.md §2](../research/logos-sadko-64m-coordination.md) | Phase 0 整体框架 |
| [phase-0-implementation-guide.md](./phase-0-implementation-guide.md) | 标准 SOP（在实施指南中已有，详细执行步骤） |
| **本文档** | **失败回退决策树 + 紧急升级**（当标准 SOP 失败时使用） |

---

## 1. 失败分类

### 1.1 三级失败定义

| 级别 | 定义 | 影响 | 处理 |
|------|------|------|------|
| **L1 轻微失败** | 单次 SOP 修复可解决 | 不影响 Phase 0 进度 | 继续 |
| **L2 中等失败** | 多次 SOP 修复失败 | 影响 Phase 0 完成时间 | 升级到协调员 |
| **L3 重大失败** | 修复不可行，需要重新设计 | 影响 Phase 1 启动 | 升级到项目负责人 |

### 1.2 失败判定的客观标准

**L1（轻微）**：
- 训练中断但有 checkpoint 可恢复
- 评估指标偏差 < 10%
- 单次 SOP 修复见效

**L2（中等）**：
- SOP 修复 3 次仍失败
- 评估指标偏差 10-30%
- 需要调整实验范围

**L3（重大）**：
- 修复不可行（如 P.0.1 MiniMind3 训练永远不收敛）
- 评估指标偏差 > 30%
- 需要重新设计实验或选择新技术栈

---

## 2. P.0.1 失败决策树

```
P.0.1 MiniMind3 64M 从零训练失败
│
├─ 失败模式 A：OOM
│  ├─ Level 1 SOP（减小 batch_size）
│  ├─ Level 2 SOP（启用 gradient_accumulation）
│  ├─ Level 3 SOP（启用 gradient_checkpointing）
│  ├─ Level 4 SOP（缩短 sequence_length）
│  ├─ 成功 → 继续
│  └─ 仍失败 → 回退方案：MiniMind3-Lite（5 层 → 3 层 + 简化 attention）
│
├─ 失败模式 B：NaN/Loss 爆炸
│  ├─ Level 1 SOP（启用 gradient clipping）
│  ├─ Level 2 SOP（降低学习率 ×0.5 → ×0.1）
│  ├─ Level 3 SOP（BF16 → FP32 切换）
│  ├─ Level 4 SOP（检查数据 pipeline 损坏）
│  └─ 仍失败 → 回退方案：换优化器（AdamW → Lion）
│
├─ 失败模式 C：Loss 不降
│  ├─ Level 1 SOP（调整 LR）
│  ├─ Level 2 SOP（增加 warmup）
│  ├─ Level 3 SOP（重新生成数据）
│  └─ 仍失败 → 回退方案：换 base（MiniMind3 → 简化 Dense）
│
└─ 失败模式 D：基础架构不可行
   ├─ 检查是否 MiniMind3 特定问题
   ├─ 试 LLaMA/Mistral 同等尺寸 base
   └─ 仍失败 → L3 升级（重新选 base）
```

---

## 3. P.0.2 失败决策树

```
P.0.2 CNN 感知层失败
│
├─ 失败模式 A：CIFAR-10 准确率低
│  ├─ Level 1 SOP（调整学习率 5e-4 → 1e-3）
│  ├─ Level 2 SOP（增加数据增强）
│  ├─ Level 3 SOP（扩展到 5 层 CNN）
│  └─ 仍失败 → 回退方案：换成 Vision Transformer (ViT) 小型版本
│
├─ 失败模式 B：对齐 cosine sim 低
│  ├─ Level 1 SOP（增加文本数据）
│  ├─ Level 2 SOP（调整对齐损失权重）
│  └─ 仍失败 → 回退方案：分开评估，CNN 单独 + 对齐单独训练
│
└─ 失败模式 C：训练崩溃
   ├─ Level 1 SOP（梯度裁剪）
   ├─ Level 2 SOP（图像预处理验证）
   └─ 仍失败 → L2 升级（图像归一化有系统性问题）
```

---

## 4. P.0.3 失败决策树

```
P.0.3 数据 Pipeline 失败
│
├─ 失败模式 A：DataLoader 卡死
│  ├─ Level 1 SOP（减少 num_workers）
│  ├─ Level 2 SOP（关闭 persistent_workers）
│  ├─ Level 3 SOP（增加 timeout）
│  └─ 仍失败 → 回退方案：单线程 DataLoader
│
├─ 失败模式 B：token 错位
│  ├─ Level 1 SOP（重新加载 tokenizer）
│  ├─ Level 2 SOP（清理数据）
│  └─ 仍失败 → L2 升级（数据源可能有 unicode 问题）
│
└─ 失败模式 C：OOM in Dataset
   ├─ Level 1 SOP（减小 dataset 大小）
   ├─ Level 2 SOP（流式加载）
   └─ 仍失败 → 回退方案：使用更小的数据子集
```

---

## 5. P.0.4 失败决策树

```
P.0.4 训练基础设施失败
│
├─ 失败模式 A：BF16 NaN
│  ├─ Level 1 SOP（启用 gradient clipping max_norm=1.0）
│  ├─ Level 2 SOP（BF16 → FP32 切换）
│  ├─ Level 3 SOP（换 Loss scaling）
│  └─ 仍失败 → 回退方案：纯 FP32 训练（牺牲速度保稳定）
│
├─ 失败模式 B：Checkpoint 损坏
│  ├─ Level 1 SOP（`weights_only=False` 加载）
│  ├─ Level 2 SOP（退回上一个 checkpoint）
│  └─ 仍失败 → 回退方案：完全重新训练（接受时间损失）
│
└─ 失败模式 C：设备错误（GPU 异常）
   ├─ Level 1 SOP（重启 GPU 驱动）
   ├─ Level 2 SOP（切换到备机）
   └─ 仍失败 → L2 升级（硬件问题，需物理干预）
```

---

## 6. P.0.5 失败决策树

```
P.0.5 评估框架失败
│
├─ 失败模式 A：评估结果不一致（同路线两次 > 5%）
│  ├─ Level 1 SOP（固定 random seed）
│  ├─ Level 2 SOP（检查 GPU throttle / 温度）
│  ├─ Level 3 SOP（重跑三次取平均）
│  └─ 仍失败 → 回退方案：用更简单的指标（如 loss 而非 acc）
│
├─ 失败模式 B：跨路线评估不可比
│  ├─ Level 1 SOP（强制要求 unified_eval.py）
│  ├─ Level 2 SOP（标记私自评估的结果为 ❌）
│  └─ 仍失败 → L2 升级（路线不配合）
│
└─ 失败模式 C：评估指标本身不适用
   └─ L2 升级（重新设计评估指标）
```

---

## 7. P.0.6 失败决策树

```
P.0.6 端侧并行化基础设施失败
│
├─ 失败模式 A：Radix Cache 命中率低
│  ├─ Level 1 SOP（检查 prefix_hash 生成）
│  ├─ Level 2 SOP（增加缓存容量）
│  └─ 仍失败 → 回退方案：用简单 LRU 缓存替代
│
├─ 失败模式 B：Per-Token 早退不收敛
│  ├─ Level 1 SOP（调整 epsilon）
│  ├─ Level 2 SOP（增加 warmup）
│  └─ 仍失败 → 回退方案：固定 K（不用早退）
│
└─ 失败模式 C：Hierarchical 难以端到端训练
   ├─ Level 1 SOP（检查主/子 block 设计）
   └─ 仍失败 → 回退方案：不用层次化，单层多路径
```

---

## 8. 紧急升级路径（Escalation Path）

### 8.1 升级流程图

```
Phase 0 任务执行
│
├─ SOP 修复成功 → 继续
│
└─ SOP 修复失败
   │
   ├─ 3 次尝试内成功 → 记录到日志，继续
   │
   └─ 3 次后仍失败
      │
      ├─ L1 升级：路线内部解决（路线负责人 + 团队）
      │  ├─ 时间预算：1 天
      │  ├─ 决策权：继续尝试 / 升级
      │  └─ 仍失败 → L2 升级
      │
      ├─ L2 升级：协调员介入（路线负责人 + 协调员）
      │  ├─ 时间预算：3 天
      │  ├─ 决策权：调整范围 / 回退方案 / L3 升级
      │  └─ 仍失败 → L3 升级
      │
      └─ L3 升级：项目负责人决定（全体参与者）
         ├─ 时间预算：1 周
         ├─ 决策权：重新设计 / 延期 / 暂停 / 转向
         └─ 输出：项目层面的决策记录
```

### 8.2 升级时机矩阵

| 严重性 | 升级时机 | 决策权 |
|--------|----------|--------|
| L1 | 3 次 SOP 失败 | 路线负责人 |
| L2 | L1 超过 1 天未解决 | 协调员 |
| L3 | L2 超过 3 天未解决 | 项目负责人 |

### 8.3 通信协议

**紧急事故响应**：

```
发现紧急事故
   ↓
立即通知（Slack #latentmind-emergency）
   ↓
15 分钟内响应（路线负责人）
   ↓
诊断 + 实施 SOP（最多 3 次尝试）
   ↓
若未解决 → L1 升级
   ↓
若仍未解决 → L2 / L3 升级
   ↓
写事故报告（Post-Incident Report）
```

---

## 9. Phase 0 综合失败场景

### 9.1 多个 P.0.x 同时失败

```
Phase 0 同时失败数 ≥ 3
│
├─ 评估是否互相影响
│  ├─ 互相影响 → L3 升级（基础设施级别问题）
│  └─ 独立失败 → 各自处理
│
└─ 时间预算评估
   ├─ 仍能在 5 周内完成 → 继续
   └─ 不能 → 延期 2 周（Phase 0 +2 周）
```

### 9.2 重大技术失败

如果遇到**架构层面**的失败（如 MiniMind3 训练永远不收敛），必须升级到 L3：

```
L3 升级流程
├─ 1. 路线负责人写《技术失败分析》
├─ 2. 召集紧急协调会议（24 小时内）
├─ 3. 决策：
│   ├─ 重新选 base 架构
│   ├─ 重新设计 Phase 0 子任务
│   ├─ 延期 Phase 0 + 1 月
│   └─ 暂停双轨，重新评估方向
├─ 4. 更新 [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md)
└─ 5. 通知所有参与者
```

---

## 10. 失败案例库（Post-Incident Reports）

待 Phase 0 启动后填充。本节记录真实失败案例与 SOP 有效性：

```
# 案例占位（待填充）
## 案例 #1: <日期> P.0.1 OOM
- 触发：训练 OOM at step 500
- SOP 1：batch_size 64 → 32 ✅ 解决
- 总结：及时见效，不需要升级
```

---

## 11. 应急联系与权限

### 11.1 联系人矩阵

| 角色 | 负责人 | 联系方式 | 决策权 |
|------|--------|---------|--------|
| **路线负责人（Logos）** | <TBD> | <Slack> | Logos 路线 L1 |
| **路线负责人（SADKO）** | <TBD> | <Slack> | SADKO 路线 L1 |
| **协调员** | <TBD> | <Slack> | L2 + 跨路线 |
| **项目负责人** | <TBD> | <Slack> | L3 + 战略 |
| **硬件运维** | <TBD> | <Slack> | GPU/系统问题 |
| **数据负责人** | <TBD> | <Slack> | 数据集问题 |

### 11.2 权限与签名

任何 SOP 修改需要：
- L1：路线负责人签名
- L2：协调员 + 路线负责人签名
- L3：项目负责人签名 + 全体通知

---

## 12. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md) | Phase 0 整体框架 |
| [phase-0-implementation-guide.md](./phase-0-implementation-guide.md) | 标准实施 SOP（在实施指南中） |
| **本文档** | **失败回退详细决策树**（当标准 SOP 失败时使用） |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Phase 0 失败回退 SOP）
**版本**：v1.0