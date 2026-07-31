AGENTS.md:135:**详细路线图**：[docs/research/logos-roadmap.md](docs/research/logos-roadmap.md)
AGENTS.md:171:> **2026-07-29 增补**：SADKO 从"文本认知推理备选"升级为"**多模态流形记忆与感知**"核心研究方向。详细论证见 [SADKO 多模态原生设计](docs/research/sadko-multimodal-native.md)。
AGENTS.md:18:> - **Logos 主线白皮书**：[`docs/research/logos-whitepaper.md`](docs/research/logos-whitepaper.md)
AGENTS.md:19:> - **Logos K 值策略**：[`docs/research/logos-k-strategy.md`](docs/research/logos-k-strategy.md)
AGENTS.md:20:> - **Logos 路线图**：[`docs/research/logos-roadmap.md`](docs/research/logos-roadmap.md)
AGENTS.md:21:> - **Logos 64M 验证计划**：[`docs/research/logos-64m-validation-plan.md`](docs/research/logos-64m-validation-plan.md)
AGENTS.md:22:> - **Logos / SADKO 双轨协调中枢**：[`docs/research/logos-sadko-64m-coordination.md`](docs/research/logos-sadko-64m-coordination.md)（Phase 0 共享前置 + 交叉验证 + 决策矩阵）
AGENTS.md:26:> - **Logos v1.0 架构（双时间尺度对比）**：[`docs/research/logos-v1-architecture.md`](docs/research/logos-v1-architecture.md)
AGENTS.md:27:> - **Logos v2.0 架构（多种循环策略）**：[`docs/research/logos-v2-architecture.md`](docs/research/logos-v2-architecture.md)
AGENTS.md:28:> - **Logos v3.0 架构（GRAM + Radix Cache）**：[`docs/research/logos-v3-architecture.md`](docs/research/logos-v3-architecture.md)
AGENTS.md:29:> - **SADKO 白皮书**：[`docs/research/sadko-whitepaper.md`](docs/research/sadko-whitepaper.md)
AGENTS.md:30:> - **SADKO 多模态原生设计**：[`docs/research/sadko-multimodal-native.md`](docs/research/sadko-multimodal-native.md)
docs/implementation/README.md:12:| `docs/research/` | 架构白皮书、路线图、协调文档 | 长期（+0 至 +12 月） | ❌ 规划/设计 |
docs/implementation/README.md:29:1. 文档必须明确标注**上游文档**（引用 `docs/research/` 中的对应设计文档）
docs/implementation/README.md:3:> **用途**：存放**近期可执行**的工程规范、实施指南、脚本模板、SOP——区别于 `docs/research/`（总体规划与架构设计）和 `docs/rfcs/`（R&D 设计提案）。
docs/references/README.md:228:**对 LatentMind 决策影响**（详见 [docs/research/README.md §0](../research/README.md#0-latentmind-双轨架构定位logos-主线--sadko-探索分支)）：
docs/research/README.md:1:# 内部研究记录索引（docs/research/）
docs/research/README.md:47:| `docs/research/` | **内部研究讨论记录**（架构推演、多轮总结的合并稿） |
docs/research/logos-64m-validation-plan.md:418:| [docs/research/sadko-v1-architecture.md](./sadko-v1-architecture.md) | SADKO Split-GQA 详细实现（v1.0 A.2 借鉴）|
docs/research/logos-roadmap.md:337:| [docs/research/sadko-64m-validation-plan.md](./sadko-64m-validation-plan.md) | SADKO 64M 验证计划（并行）|
docs/research/logos-roadmap.md:338:| [docs/research/sadko-whitepaper.md](./sadko-whitepaper.md) | SADKO 白皮书（双轨协同）|
docs/research/logos-v1-architecture.md:498:| [docs/research/hrm-text.md](../references/hrm-text.md) | HRM-Text 原论文笔记（仅参考） |
docs/research/logos-whitepaper.md:399:| [docs/research/sadko-whitepaper.md](./sadko-whitepaper.md) | SADKO 探索分支白皮书（互补关系）|
docs/research/logos-whitepaper.md:400:| [docs/research/sadko-multimodal-native.md](./sadko-multimodal-native.md) | 双轨分工的第一性原理论证 |
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1000:# AGENTS.md 中可能引用 docs/research/logos-*.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1001:sed -i 's|docs/research/logos-whitepaper\.md|docs/research/logos/whitepaper.md|g; s|docs/research/logos-k-strategy\.md|docs/research/logos/k-strategy.md|g; s|docs/research/logos-roadmap\.md|docs/research/logos/roadmap.md|g; s|docs/research/logos-64m-validation-plan\.md|docs/research/logos/64m-validation-plan.md|g; s|docs/research/logos-v1-architecture\.md|docs/research/logos/v1-architecture.md|g; s|docs/research/logos-v2-architecture\.md|docs/research/logos/v2-architecture.md|g; s|docs/research/logos-v3-architecture\.md|docs/research/logos/v3-architecture.md|g; s|docs/research/logos-sadko-64m-coordination\.md|docs/research/logos/sadko-64m-coordination.md|g' AGENTS.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1011:sed -i 's|docs/research/sadko-whitepaper\.md|docs/research/sadko/whitepaper.md|g; s|docs/research/sadko-64m-validation-plan\.md|docs/research/sadko/64m-validation-plan.md|g; s|docs/research/sadko-v1-architecture\.md|docs/research/sadko/v1-architecture.md|g; s|docs/research/sadko-v2-architecture\.md|docs/research/sadko/v2-architecture.md|g; s|docs/research/sadko-v3-architecture\.md|docs/research/sadko/v3-architecture.md|g; s|docs/research/sadko-multimodal-native\.md|docs/research/sadko/multimodal-native.md|g; s|docs/research/sadko-open-issues\.md|docs/research/sadko/open-issues.md|g; s|docs/research/sadko-elf-vs-gdm-review\.md|docs/research/sadko/elf-vs-gdm-review.md|g; s|docs/research/sadko-elf-graph-emergence\.md|docs/research/sadko/elf-graph-emergence.md|g; s|docs/research/sadko-elf-phase0-manual\.md|docs/research/sadko/elf-phase0-manual.md|g; s|docs/research/sadko-elf-lifecycle\.md|docs/research/sadko/elf-lifecycle.md|g' AGENTS.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1021:grep -E "docs/research/(sadko-|logos-)[a-z]" AGENTS.md || echo "OK: AGENTS.md 无旧前缀引用残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1026:- [ ] **Step 5: 检查并更新其他可能引用 docs/research/ 的文件**
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1030:# 扫描所有可能引用 docs/research/ 的文件
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1031:grep -rln "docs/research/" docs/ README.md 2>/dev/null | grep -v "docs/research/REFACTOR_MAP" | grep -v "docs/research/README" | grep -v "docs/research/sadko/" | grep -v "docs/research/logos/" | grep -v "docs/research/elf/"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1034:Expected: 输出任何引用 docs/research/ 旧路径的文件清单（如有，逐个处理）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1073:grep -rn "docs/research/sadko-\|docs/research/logos-" . --include="*.md" 2>/dev/null | grep -v "docs/research/sadko/sadko-64m-coordination.md" | grep -v "docs/research/logos/sadko-64m-coordination.md" || echo "OK: 全项目无旧前缀引用残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:107:git mv docs/research/sadko-whitepaper.md docs/research/sadko/whitepaper.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1083:ls docs/research/sadko/*.md | wc -l
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1086:ls docs/research/logos/*.md | wc -l
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1089:ls docs/research/elf/*.md | wc -l
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:108:git mv docs/research/sadko-64m-validation-plan.md docs/research/sadko/64m-validation-plan.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1091:echo "=== 顶级 docs/research/ 文件数（应仅含 README.md + REFACTOR_MAP.md）==="
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1092:ls docs/research/*.md | wc -l
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1093:ls docs/research/*.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:109:git mv docs/research/sadko-v1-architecture.md docs/research/sadko/v1-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1108:grep -E "\.\./sadko/elf-[a-z-]+\.md" docs/research/elf/README.md | head -3 | while read line; do
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:110:git mv docs/research/sadko-v2-architecture.md docs/research/sadko/v2-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1110:  full_path="docs/research/elf/$target"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1117:grep -E "\./(sadko|logos|elf)/README\.md" docs/research/README.md | while read line; do
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1119:  full_path="docs/research/$target"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:111:git mv docs/research/sadko-v3-architecture.md docs/research/sadko/v3-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:112:git mv docs/research/sadko-multimodal-native.md docs/research/sadko/multimodal-native.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1130:echo "=== docs/research/ 最终结构 ==="
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1131:tree docs/research/ -L 2 2>/dev/null || find docs/research/ -maxdepth 2 -type f | sort
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:113:git mv docs/research/sadko-open-issues.md docs/research/sadko/open-issues.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1141:echo "" >> docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1142:echo "## 验证完成" >> docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1143:echo "" >> docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1144:echo "2026-07-30 验证完成：所有跨文档引用已更新为新子目录路径。" >> docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1145:git add docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:114:git mv docs/research/sadko-elf-vs-gdm-review.md docs/research/sadko/elf-vs-gdm-review.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1155:- [ ] `docs/research/sadko/` 含 11 份原文档 + 1 个 README.md（12 个 .md 文件）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1156:- [ ] `docs/research/logos/` 含 8 份原文档 + 1 个 README.md（9 个 .md 文件）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1157:- [ ] `docs/research/elf/` 含 1 个 README.md（含胼胝体契约）+ 3 个骨架文件
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1158:- [ ] 顶级 `docs/research/` 仅含 README.md + REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:115:git mv docs/research/sadko-elf-graph-emergence.md docs/research/sadko/elf-graph-emergence.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:1160:- [ ] AGENTS.md 中所有 docs/research/ 引用已更新
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:116:git mv docs/research/sadko-elf-phase0-manual.md docs/research/sadko/elf-phase0-manual.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:117:git mv docs/research/sadko-elf-lifecycle.md docs/research/sadko/elf-lifecycle.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:118:ls docs/research/sadko/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:127:ls docs/research/ | grep -E "^sadko-" || echo "OK: 无 sadko-* 文件残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:138:sed -i 's|sadko-whitepaper\.md|whitepaper.md|g; s|sadko-64m-validation-plan\.md|64m-validation-plan.md|g; s|sadko-v1-architecture\.md|v1-architecture.md|g; s|sadko-v2-architecture\.md|v2-architecture.md|g; s|sadko-v3-architecture\.md|v3-architecture.md|g; s|sadko-multimodal-native\.md|multimodal-native.md|g; s|sadko-open-issues\.md|open-issues.md|g; s|sadko-elf-vs-gdm-review\.md|elf-vs-gdm-review.md|g; s|sadko-elf-graph-emergence\.md|elf-graph-emergence.md|g; s|sadko-elf-phase0-manual\.md|elf-phase0-manual.md|g; s|sadko-elf-lifecycle\.md|elf-lifecycle.md|g' docs/research/sadko/*.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:149:grep -rn "sadko-whitepaper\|sadko-64m-validation\|sadko-v[123]-architecture\|sadko-multimodal\|sadko-open-issues\|sadko-elf-" docs/research/sadko/ || echo "OK: 无 SADKO 旧前缀引用残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:158:git add docs/research/sadko/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:175:git mv docs/research/logos-whitepaper.md docs/research/logos/whitepaper.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:176:git mv docs/research/logos-k-strategy.md docs/research/logos/k-strategy.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:177:git mv docs/research/logos-roadmap.md docs/research/logos/roadmap.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:178:git mv docs/research/logos-64m-validation-plan.md docs/research/logos/64m-validation-plan.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:179:git mv docs/research/logos-v1-architecture.md docs/research/logos/v1-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:180:git mv docs/research/logos-v2-architecture.md docs/research/logos/v2-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:181:git mv docs/research/logos-v3-architecture.md docs/research/logos/v3-architecture.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:182:git mv docs/research/logos-sadko-64m-coordination.md docs/research/logos/sadko-64m-coordination.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:183:ls docs/research/logos/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:18:- `docs/research/sadko/README.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:192:ls docs/research/ | grep -E "^logos-" || echo "OK: 无 logos-* 文件残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:19:- `docs/research/logos/README.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:202:sed -i 's|logos-whitepaper\.md|whitepaper.md|g; s|logos-k-strategy\.md|k-strategy.md|g; s|logos-roadmap\.md|roadmap.md|g; s|logos-64m-validation-plan\.md|64m-validation-plan.md|g; s|logos-v1-architecture\.md|v1-architecture.md|g; s|logos-v2-architecture\.md|v2-architecture.md|g; s|logos-v3-architecture\.md|v3-architecture.md|g' docs/research/logos/*.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:20:- `docs/research/elf/README.md`（核心：含胼胝体契约五元组 + 五个不变量）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:212:grep -rn "logos-whitepaper\|logos-k-strategy\|logos-roadmap\|logos-64m-validation\|logos-v[123]-architecture" docs/research/logos/ || echo "OK: 无 Logos 旧前缀引用残留"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:21:- `docs/research/elf/memory-architecture.md`（骨架）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:221:git add docs/research/logos/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:22:- `docs/research/elf/graph-growth.md`（骨架）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:230:- Create: `docs/research/elf/README.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:234:写入以下内容到 `docs/research/elf/README.md`（含 §0 定位 + §1 三子方向索引 + §2 胼胝体契约五元组 + §3 五个不变量 + §4 演进路径 + §5 与 sadko/elf-* 文档的关系）：
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:237:cat > /workspace/project/LatentMind/docs/research/elf/README.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:23:- `docs/research/elf/retrieval-extraction.md`（骨架）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:26:SADKO（11 份）→ `docs/research/sadko/`：
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:29:Logos（8 份）→ `docs/research/logos/`：
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:33:- `docs/research/README.md`（重写为三研究线索引）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:35:- `docs/architecture.md`（如引用了 docs/research/ 路径）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:36:- `docs/implementation/README.md`（如引用了 docs/research/ 路径）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:404:[ -f docs/research/elf/README.md ] && echo "OK: README.md 存在" || echo "FAIL"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:406:grep -q "## 2. 胼胝体接口契约" docs/research/elf/README.md && echo "OK: §2 契约章节存在" || echo "FAIL: §2 契约章节缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:407:grep -q "### 2.2 接口契约五元组" docs/research/elf/README.md && echo "OK: §2.2 五元组存在" || echo "FAIL: §2.2 五元组缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:408:grep -q "### 2.3 关键不变量" docs/research/elf/README.md && echo "OK: §2.3 不变量存在" || echo "FAIL: §2.3 不变量缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:418:grep -E "\.\./sadko/elf-" docs/research/elf/README.md | head -5
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:419:grep -E "\./memory-architecture\.md" docs/research/elf/README.md | head -3
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:428:git add docs/research/elf/README.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:437:- Create: `docs/research/elf/memory-architecture.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:438:- Create: `docs/research/elf/graph-growth.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:439:- Create: `docs/research/elf/retrieval-extraction.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:43:- Create: `docs/research/sadko/`、`docs/research/logos/`、`docs/research/elf/`（空目录）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:444:cat > /workspace/project/LatentMind/docs/research/elf/memory-architecture.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:44:- Create: `docs/research/REFACTOR_MAP.md`（迁移清单 + 路径映射表）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:49:mkdir -p /workspace/project/LatentMind/docs/research/sadko
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:50:mkdir -p /workspace/project/LatentMind/docs/research/logos
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:517:cat > /workspace/project/LatentMind/docs/research/elf/graph-growth.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:51:mkdir -p /workspace/project/LatentMind/docs/research/elf
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:52:ls -la /workspace/project/LatentMind/docs/research/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:57:- [ ] **Step 2: 扫描所有引用 docs/research/ 的文件**
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:590:cat > /workspace/project/LatentMind/docs/research/elf/retrieval-extraction.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:5:**Goal:** 重构 `docs/research/` 为 sadko/logos/elf 三子目录结构，迁移 19 份现有文档并创建 `elf/README.md`（含 §2 胼胝体接口契约 + 五个不变量），为 ELF 独立研究线建立稳定基础设施。
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:61:grep -rn "docs/research/" docs/ AGENTS.md README.md 2>/dev/null | grep -v "^Binary" | sort -u
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:64:Expected: 输出引用 docs/research/ 下文件的所有位置（包括文件名 + 行号 + 引用内容）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:670:  [ -f "docs/research/elf/$f" ] && echo "OK: $f 存在" || echo "FAIL: $f 缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:671:  grep -q "## 0. 核心问题" "docs/research/elf/$f" && echo "OK: $f §0 存在" || echo "FAIL: $f §0 缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:672:  grep -q "## 6. 鲁棒性评分" "docs/research/elf/$f" && echo "OK: $f §6 存在" || echo "FAIL: $f §6 缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:682:git add docs/research/elf/memory-architecture.md docs/research/elf/graph-growth.md docs/research/elf/retrieval-extraction.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:691:- Create: `docs/research/sadko/README.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:692:- Create: `docs/research/logos/README.md`
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:697:cat > /workspace/project/LatentMind/docs/research/sadko/README.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:72:grep -rn "docs/research/" docs/ AGENTS.md README.md 2>/dev/null | grep -v "^Binary" | sort -u > docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:73:wc -l docs/research/REFACTOR_MAP.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:762:cat > /workspace/project/LatentMind/docs/research/logos/README.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:7:**Architecture:** Phase A：迁移现有 docs/research/ 文件到三子目录（含路径去前缀 + 跨文档引用更新）。Phase B：创建 elf/ 子目录骨架（含 README.md 完整胼胝体契约 + 三子方向骨架文件）。Phase C：重写顶级 docs/research/README.md 为三研究线索引。Phase D：更新 AGENTS.md 等外部引用。
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:826:  [ -f "docs/research/$subdir/README.md" ] && echo "OK: $subdir/README.md 存在" || echo "FAIL: $subdir/README.md 缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:827:  grep -q "## 1. 文件清单" "docs/research/$subdir/README.md" && echo "OK: $subdir §1 存在" || echo "FAIL: $subdir §1 缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:82:[ -d docs/research/sadko ] && [ -d docs/research/logos ] && [ -d docs/research/elf ] && [ -s docs/research/REFACTOR_MAP.md ] && echo "OK: 三子目录 + REFACTOR_MAP.md 就绪" || echo "FAIL"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:837:git add docs/research/sadko/README.md docs/research/logos/README.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:843:### Task 7: 重写顶级 docs/research/README.md 为三研究线索引
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:846:- Modify: `docs/research/README.md`（重写）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:852:cp docs/research/README.md /tmp/research-readme-backup.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:862:cat > /workspace/project/LatentMind/docs/research/README.md << 'EOF'
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:863:# 内部研究记录索引（docs/research/）
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:888:docs/research/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:91:git add docs/research/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:936:echo "顶级 docs/research/README.md 已重写"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:945:grep -q "Logos 主线" docs/research/README.md && echo "OK: Logos 入口存在" || echo "FAIL: Logos 入口缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:946:grep -q "SADKO 主干" docs/research/README.md && echo "OK: SADKO 入口存在" || echo "FAIL: SADKO 入口缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:947:grep -q "ELF 独立研究线" docs/research/README.md && echo "OK: ELF 入口存在" || echo "FAIL: ELF 入口缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:948:grep -q "REFACTOR_MAP" docs/research/README.md && echo "OK: REFACTOR_MAP 引用存在" || echo "FAIL: REFACTOR_MAP 引用缺失"
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:958:ls docs/research/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:961:ls docs/research/sadko/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:964:ls docs/research/logos/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:967:ls docs/research/elf/
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:976:git add docs/research/README.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:987:- [ ] **Step 1: 扫描 AGENTS.md 中所有 docs/research/ 引用**
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:991:grep -n "docs/research/" AGENTS.md
docs/superpowers/plans/2026-07-30-elf-research-line-foundation.md:994:Expected: 输出 AGENTS.md 中所有 docs/research/ 引用位置（行号 + 内容）
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:108:需更新所有引用 `docs/research/` 下文件的路径。重点位置：
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:113:| `docs/research/sadko/whitepaper.md` 内部 | 相对路径引用（多处，§三、§六）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:114:| `docs/research/sadko/multimodal-native.md` | 相对路径（§0、§5、§6、§8）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:115:| `docs/research/sadko/elf-*.md` | 相对路径（多处）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:116:| `docs/research/sadko/v1-v2-v3-architecture.md` | 相对路径（多处）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:117:| `docs/research/sadko/64m-validation-plan.md` | 相对路径（多处）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:118:| `docs/research/logos/*.md` | 相对路径（多处）|
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:119:| `docs/research/logos/sadko-64m-coordination.md` | 引用 `../implementation/phase-0-*.md` |
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:124:1. `grep -rn "docs/research/" docs/ AGENTS.md` 找出所有引用
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:127:4. 在 git commit 中明确标注"docs/research/ 目录重构"以便回溯
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:129:### 1.4 顶级 `docs/research/README.md` 重写
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:403:1. **子目录中 README.md 的具体内容**：每个子目录的 README 需要把原 docs/research/README.md §3.1/§3.2/§3.3 的对应部分迁移过来——具体哪些内容放 README、哪些放子文件待定
docs/superpowers/specs/2026-07-30-elf-research-line-design.md:44:docs/research/

## 验证完成

2026-07-30 验证完成：所有跨文档引用已更新为新子目录路径。

---

## 2026-07-31：ELF → Hippo 重命名（保留历史快照，上方 L1-L169 为 2026-07-30 原貌）

> **背景**：ELF（"Entity-Linked Flow"或未展开的技术缩写）作为研究线名与项目 logos/sadko 隐喻命名风格不一致。决策：**研究线名重命名为 Hippo**（海马体，hippocampus 缩写，与"右脑=海马体"项目隐喻对齐；字符数 5 与 logos/sadko 对称）。
> **历史记录保留**：上方 L1-L169 的 grep 输出为 2026-07-30 重构的真实历史快照——保留原貌以便回溯 sed/grep 命令的原始目标路径。
> **本次变更范围**（不影响历史快照）：
>
> | 类别 | 操作 | 文件数 |
> |------|------|--------|
> | 目录重命名 | `docs/research/elf/` → `docs/research/hippo/` | 1 目录 / 4 文件 |
> | 研究线文档改名 | `hippo/README.md`、3 份骨架 | 4 文件 |
> | 顶级索引更新 | `docs/research/README.md` ELF 引用 → Hippo | 1 文件 |
> | 待续 Phase 2 | sadko/elf-*.md → sadko/hippo-*.md + 主体文档 | ~12 文件 |
> | 待续 Phase 3 | docs/superpowers/{specs,plans}/ 中 elf-* 文件名 + 全文 | 5 文件 + 4 文件 |
> | 待续 Phase 3 | docs/references/implementation/rfcs 中 ELF 引用 | ~5 文件 |
>
> **后续验证点**：
> - Phase 1 完成后：`docs/research/elf/` 不应存在；`docs/research/hippo/` 应含 4 文件
> - Phase 2 完成后：`docs/research/sadko/elf-*.md` 不应存在；`docs/research/sadko/hippo-*.md` 应含 5 文件
> - Phase 3 完成后：全局 grep `ELF` 仅在 (a) REFACTOR_MAP.md 历史快照 (b) hippo/README.md "原名 ELF" 注释 等"历史别名"位置残留
>
> **哲学依据**：项目已存在"右脑=海马体"隐喻（见 AGENTS.md §7 "海马体（流形记忆）"），ELF 命名其实是"补齐一致性"而非"引入新概念"。Hippo（5 字符）与 Logos（5 字符）/ SADKO（5 字符）字符数对称。
