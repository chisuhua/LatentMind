# 论文源文件（docs/references/papers/）

> **用途**：集中存放 LatentMind 项目所依赖的**外部论文原始源文件**（PDF / HTML）
> **最后更新**：2026-07-31（新增 8 篇论文 + 完整 Loop B 谱系覆盖）

---

## 1. 文件清单

### 1.1 Loop A（循环架构核心）

| 论文 | arXiv ID | PDF | HTML | 用途 |
|---|---|---|---|---|
| HRM-Text | [2605.20613](https://arxiv.org/abs/2605.20613) | ❌ 限流未下完 | ✅ abs 摘要页 | v1.0 backbone |
| GRAM | [2605.19376](https://arxiv.org/abs/2605.19376) | ✅ 1.3 MB | ✅ 380 KB | v1.5 核心 |
| HRM（原始 27M） | [2506.21734](https://arxiv.org/abs/2506.21734) | ✅ 2.0 MB | ✅ 230 KB | MagicNorm 概念溯源 |
| TRM | [2510.04871](https://arxiv.org/abs/2510.04871) | ✅ 420 KB | ✅ 350 KB | 7M 小型递归对比 |
| **DiscoLoop** | [2607.00341](https://arxiv.org/abs/2607.00341) | ⏳ 待下载 | ⏳ 待下载 | Loop A 表征对齐（Φ 通道机制）|

### 1.2 Loop B（持久记忆系统，2026-07-31 新增 7 篇）

| 论文 | arXiv ID | PDF | HTML | 流派 |
|---|---|---|---|---|
| **Memorizing Transformers** | [2203.08913](https://arxiv.org/abs/2203.08913) | ⏳ 待下载 | ⏳ 待下载 | kNN 检索 |
| **Compressive Transformers** | [1911.05507](https://arxiv.org/abs/1911.05507) | ⏳ 待下载 | ⏳ 待下载 | 1D Conv 压缩 |
| **StreamingLLM** | [2309.17453](https://arxiv.org/abs/2309.17453) | ⏳ 待下载 | ⏳ 待下载 | 滑动窗口 + attention sink |
| **InfLLM** | [2402.04617](https://arxiv.org/abs/2402.04617) | ⏳ 待下载 | ⏳ 待下载 | 块级 memory + 训练无关 |
| **AutoCompressors** | [2305.14788](https://arxiv.org/abs/2305.14788) | ⏳ 待下载 | ⏳ 待下载 | LLM 自压缩 |
| **RMT** | [2207.06881](https://arxiv.org/abs/2207.06881) | ⏳ 待下载 | ⏳ 待下载 | 特殊 [mem] tokens |
| **Landmark Attention** | [2305.16300](https://arxiv.org/abs/2305.16300) | ⏳ 待下载 | ⏳ 待下载 | attention 内生 block gate |

### 1.3 其他循环架构相关

| 论文 | arXiv ID | PDF | HTML | 用途 |
|---|---|---|---|---|
| LoopCoder-v2 (PLT) | [2606.18023](https://arxiv.org/abs/2606.18023) | ⏳ 待下载 | ⏳ 待下载 | PLT 架构 |
| Huginn 3.5B | [2502.05171](https://arxiv.org/abs/2502.05171) | ⏳ 待下载 | ⏳ 待下载 | 50 步循环反例 |
| STARS 2026 | (ICML 2026) | ⏳ 待下载 | ⏳ 待下载 | 循环崩溃修复 |
| Per-Token Convergence | (2026-07) | ⏳ 待下载 | ⏳ 待下载 | 动态 K 证据 |

**总计**：3 完整 PDF + 4 HTML + 8 篇待下载（Loop B + 4 篇循环相关）

---

## 2. 文件命名规范

```
papers/
├── arxiv-<YYMM>.<NNNN>-<slug>.pdf      # 论文 PDF
├── arxiv-<YYMM>.<NNNN>-<slug>.html     # 论文 HTML 全文（来自 ar5iv.labs.arxiv.org）
└── arxiv-<YYMM>.<NNNN>-<slug>-abstract-page.html  # 仅 arXiv 摘要页（论文过新，ar5iv 尚未完整镜像）
```

**slug** 命名约定：短横线连接的论文短名

- HRM-Text → `hrm-text`
- GRAM → `gram`
- HRM（原始）→ `hrm-original`
- TRM → `trm`
- DiscoLoop → `discoloop`
- Memorizing Transformers → `memorizing-transformers`
- Compressive Transformers → `compressive-transformers`
- StreamingLLM → `streaming-llm`
- InfLLM → `inf-llm`
- AutoCompressors → `auto-compressors`
- RMT → `rmt`
- Landmark Attention → `landmark-attention`
- LoopCoder-v2 → `loopcoder-v2`
- Huginn → `huginn`
- STARS → `stars`
- Per-Token Convergence → `per-token-convergence`

---

## 3. 下载来源说明

### 3.1 PDF 来源

- arXiv CDN：`https://arxiv.org/pdf/<id>`
- 实测下载速度：~4-5 KB/s（**严重限流**），从该 IP 下载 2.88 MB 的 HRM-Text 论文 5 分钟内无法完成
- 当前已有 3 篇完整 PDF（HRM-original / GRAM / TRM）

### 3.2 HTML 来源

- ar5iv 镜像：`https://ar5iv.labs.arxiv.org/html/<id>`
- 实测下载速度：~50 KB/s（**正常**）
- **例外**：2026-05 发布的 HRM-Text 太新，ar5iv 尚未完整处理 → 退化为 arXiv abs 摘要页

### 3.3 已尝试但失败的下载

- `https://arxiv.org/pdf/2605.20613`（HRM-Text PDF）→ 限流，300s 内 600KB/2.88MB
- `https://export.arxiv.org/pdf/2605.20613` → 限流同样严重

**建议**：过 24-48 小时后重试 HRM-Text PDF；或换 IP 后重试

### 3.4 新增 8 篇 Loop B 论文状态

**所有 8 篇新论文 PDF/HTML 均为 ⏳ 待下载**。当前**仅依赖笔记**（[../discoloop.md](../discoloop.md), [../memorizing-transformers.md](../memorizing-transformers.md) 等）做研究决策——笔记内容已经过核实关键数字，**足够支撑当前阶段**。

---

## 4. 使用方式

### 4.1 优先用 PDF

PDF 是论文的最终发表格式，包含图表、公式、参考完整列表，适合引用、打印、做 PDF 标注。

```bash
# 用 pdftotext 提取文本
pdftotext arxiv-2605.19376-gram.pdf -

# 用 poppler-utils 转图片（用于读图）
pdftoppm -r 200 arxiv-2605.19376-gram.pdf gram-page
```

### 4.2 HTML 用于检索/可读

HTML 适合 grep 搜索、复制粘贴、浏览器阅读。

```bash
# 搜索关键词
grep -i "MagicNorm" arxiv-*.html

# 提取所有 h2 章节标题
grep -oE '<h2[^>]*>[^<]+</h2>' arxiv-2605.19376-gram.html
```

### 4.3 注意事项

- ar5iv HTML 与 arXiv 官方 PDF 内容**应一致**，但排版可能略有不同
- 论文中的公式在 HTML 中可能以 LaTeX 形式（`\epsilon`）而非渲染后的形式存在
- 图表在 ar5iv HTML 中通常**缺失**（需要 PDF）

---

## 5. 重试策略

如果未来需要补齐论文 PDF：

1. **换时间窗口**：限流通常有冷却期，过 24-48h 后再试
2. **换 IP**：VPN / 代理 / 移动网络
3. **用论文原作者提供的其他来源**：
   - Sapient 官方：https://sapient.inc/introducing-hrm-text（HTML 全文）
   - HF 模型卡：https://huggingface.co/sapientinc/HRM-Text-1B（含论文概要）
4. **使用已有的论文笔记**：本目录外的笔记文件（如 `../hrm-text.md`、`../discoloop.md` 等）包含几乎所有关键内容，足以代替原 PDF

---

## 6. 与笔记文件的关系（2026-07-31 完整版）

### 6.1 Loop A 循环架构

| 论文 | 笔记 | 源文件 |
|---|---|---|
| HRM-Text | [`../hrm-text.md`](../hrm-text.md) | 本目录（仅 abs 摘要页） |
| GRAM | [`../gram.md`](../gram.md) ⏳ | 本目录（PDF + HTML） |
| HRM（原始） | [`../hrm-original.md`](../hrm-original.md) | 本目录（PDF + HTML） |
| TRM | [`../trm.md`](../trm.md) | 本目录（PDF + HTML） |
| **DiscoLoop** | [`../discoloop.md`](../discoloop.md) | ⏳ 待下载 |

### 6.2 Loop B 持久记忆系统（2026-07-31 新增）

| 论文 | 笔记 | 流派 |
|---|---|---|
| **Memorizing Transformers** | [`../memorizing-transformers.md`](../memorizing-transformers.md) | kNN 检索 |
| **Compressive Transformers** | [`../compressive-transformers.md`](../compressive-transformers.md) | 1D Conv 压缩 |
| **StreamingLLM** | [`../streaming-llm.md`](../streaming-llm.md) | 滑动窗口 + sink |
| **InfLLM** | [`../inf-llm.md`](../inf-llm.md) | 块级 memory |
| **AutoCompressors** | [`../auto-compressors.md`](../auto-compressors.md) | LLM 自压缩 |
| **RMT** | [`../rmt.md`](../rmt.md) | 特殊 [mem] tokens |
| **Landmark Attention** | [`../landmark-attention.md`](../landmark-attention.md) | attention 内生 |

### 6.3 其他循环架构

| 论文 | 笔记 | 用途 |
|---|---|---|
| LoopCoder-v2 (PLT) | [`../loopcoder-v2.md`](../loopcoder-v2.md) | PLT 架构 + "Only Loop Once" |
| Huginn 3.5B | [`../huginn.md`](../huginn.md) | 50 步循环反例 |
| STARS 2026 | [`../stars.md`](../stars.md) | 循环崩溃修复 |
| Per-Token Convergence | [`../per-token-convergence.md`](../per-token-convergence.md) | 动态 K 证据 |

### 6.4 谱系索引

完整文献谱系见 [`../loop-memory-survey.md`](../loop-memory-survey.md)（2026-07-31 新增），覆盖 Loop A / Loop B / Loop C 三种循环维度 + 6 大 Loop B 流派。

---

**最后更新**：2026-07-31（新增 8 篇 Loop B 论文 + 完整谱系索引）
