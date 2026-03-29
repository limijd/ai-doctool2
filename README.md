# aidoc - AI 文档处理工具链

将 PDF 文档转换为高质量 Markdown，并为 RAG（检索增强生成）应用构建语义索引。

专为技术标准文档（IEEE、ISO 等）、论文、技术白皮书等复杂文档优化。

---

## 工具链概览

```
PDF Input
    |
    v
[1] aidoc_convert --------- PDF -> Markdown (Docling)
    |                       OCR / table / formula / code
    v
[2] aidoc_strip ----------- Strip headers & footers
    |                       statistics -> heuristics -> LLM -> merge
    v
[3] aidoc_fix_hierarchy --- Fix heading levels
    |                       numbering rules + interval + LLM
    v
[4] aidoc_fix_codeblocks -- Fix code block boundaries
    |                       prose/code detection + fence repair
    v
[5] aidoc_index ----------- Build RAG semantic index
    |                       chunking / summary / keywords -> JSON
    v
RAG-ready Markdown + JSON Index
```

每个工具**独立可用**，也可以按顺序串联使用。

---

## 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

如果不使用 PDF 转换功能（只处理已有的 Markdown），只需 `pip3 install requests`。

### 2. 配置 LLM（可选）

工具链支持两种 LLM 后端，通过 `aidoc.conf` 统一配置：

**Ollama（本地部署，推荐）：**

```bash
# 安装 Ollama: https://ollama.ai
ollama pull qwen3:8b

# 默认配置即可，无需修改 aidoc.conf
```

**OpenAI API：**

```bash
cp aidoc.openai.example.conf aidoc.conf
# 编辑 aidoc.conf，填入 api_key
```

所有工具也支持 `--no-llm` 参数，不依赖任何 LLM 即可运行（使用纯规则引擎）。

### 3. 处理文档

```bash
# 完整流水线
python3 aidoc_convert.py document.pdf
python3 aidoc_strip.py document.md
python3 aidoc_fix_hierarchy.py document_clean.md
python3 aidoc_fix_codeblocks.py document_clean.md
python3 aidoc_index.py document_clean.md

# 快速模式（不使用 LLM）
python3 aidoc_strip.py document.md --no-llm
python3 aidoc_fix_hierarchy.py document.md --no-llm
python3 aidoc_index.py document.md --no-llm
```

---

## 工具详解

### aidoc_convert.py — PDF 转 Markdown

基于 [Docling](https://github.com/DS4SD/docling) 的高质量 PDF 转换器。

**核心能力：**
- 代码块自动识别与语言标注
- LaTeX 公式提取
- 高精度表格提取（TableFormer ACCURATE 模式）
- 跨页代码块自动合并
- 页眉页脚初步过滤
- OCR 支持（扫描文档）
- GPU 加速（CUDA/MPS）

```bash
# 基本用法
python3 aidoc_convert.py input.pdf

# 指定输出
python3 aidoc_convert.py input.pdf -o output.md

# 快速表格模式
python3 aidoc_convert.py input.pdf --table-mode fast

# GPU 加速
python3 aidoc_convert.py input.pdf --device cuda

# 禁用 OCR（纯文本 PDF）
python3 aidoc_convert.py input.pdf --no-ocr
```

---

### aidoc_strip.py — 页眉页脚剥离

四层智能检测架构，通用化设计（不 hardcode 特定文档内容）。

**检测层级：**

| 层级 | 方法 | 说明 |
|------|------|------|
| Layer 1 | 统计模式检测 | 高频重复行 + 归一化 + 相似度聚类 |
| Layer 2 | 启发式分类 | 页码/页眉/页脚/水印 结构特征 |
| Layer 3 | LLM 验证 | 仅对中等置信度模式触发 |
| Layer 3.5 | 代码块清理 | 代码块内残留的页眉页脚 |
| Layer 4 | 内容合并 | 修复被分割的代码块和表格 |

**置信度系统：**
- `≥0.6` 高置信度：直接移除
- `[0.35, 0.6)` 中置信度：触发 LLM 验证
- `<0.35` 低置信度：丢弃

```bash
# 完整模式（含 LLM 验证）
python3 aidoc_strip.py input.md

# 快速模式
python3 aidoc_strip.py input.md --no-llm

# 交互确认
python3 aidoc_strip.py input.md --interactive

# 预览（不写文件）
python3 aidoc_strip.py input.md --dry-run
```

**输出文件：**
- `input_clean.md` — 清理后的 Markdown
- `input.removed.txt` — 被移除的行及原因
- `input.cleanup.txt` — 合并操作日志

---

### aidoc_fix_hierarchy.py — 标题层级修复

修复 PDF 转换后标题层级退化（如所有标题都变成 `##`）的问题。

**五阶段算法：**

1. **页眉污染检测** — 识别并移除每页重复的文档标题
2. **规则骨架构建** — 基于编号模式推断层级（`1.` → H2, `1.1` → H3, ...）
3. **区间归属分析** — 无编号标题归入 `[前编号, 后编号)` 区间
4. **内联小节处理** — Rules、Permissions、Example 等继承父级 +1
5. **LLM 辅助推断** — 处理剩余无编号标题

```bash
# 基本用法（覆盖原文件）
python3 aidoc_fix_hierarchy.py document.md

# 预览
python3 aidoc_fix_hierarchy.py document.md --dry-run

# 纯规则模式
python3 aidoc_fix_hierarchy.py document.md --no-llm

# 输出报告
python3 aidoc_fix_hierarchy.py document.md --report report.json
```

---

### aidoc_fix_codeblocks.py — 代码块边界修复

修复 PDF 转换后代码块 ``` 标记错位导致的问题。

**处理的问题类型：**
- 正文被错误包含在代码块内
- 缩进的 ``` 被当作代码内容
- 未闭合的代码块
- 代码块边界混乱

**智能识别：**
- 正文特征：Markdown 标题、长段落、表格
- 代码特征：VHDL/Verilog/SV/C/Python 关键字、语法符号
- 置信度分级：HIGH / MEDIUM / LOW

```bash
# 分析并修复
python3 aidoc_fix_codeblocks.py input.md

# 仅分析
python3 aidoc_fix_codeblocks.py input.md --dry-run

# 详细报告
python3 aidoc_fix_codeblocks.py input.md -v --dry-run
```

---

### aidoc_index.py — RAG 语义索引

将 Markdown 文档切片并生成结构化 JSON 索引，用于 RAG 检索。

**核心能力：**
- 基于标题层级的语义切片
- LLM 生成章节摘要和关键字
- 自动深度调节（根据文件大小）
- 层次化目录树（TOC Tree）
- 关键字倒排索引

**自动切分策略：**

| 文件大小 | 切分深度 | 说明 |
|----------|----------|------|
| < 50KB | H1-H4 | 细粒度 |
| 50-200KB | H1-H3 | 中等 |
| > 200KB | H1-H2 | 粗粒度 |

```bash
# 完整索引（含 LLM 摘要）
python3 aidoc_index.py document.md

# 纯结构索引（无 LLM）
python3 aidoc_index.py document.md --no-llm

# 强制切分深度
python3 aidoc_index.py document.md --depth 3

# 指定输出
python3 aidoc_index.py document.md -o index.json
```

**索引格式示例：**

```json
{
  "source_file": "document.md",
  "file_size": 123456,
  "total_lines": 2000,
  "depth_level": 3,
  "toc_tree": { "id": "root", "children": [...] },
  "chunks": {
    "chunk_001": {
      "title": "1. Overview",
      "level": 2,
      "start_line": 10,
      "end_line": 150,
      "summary": "本章概述了...",
      "keywords": ["overview", "scope", "architecture"]
    }
  },
  "keyword_index": {
    "architecture": ["chunk_001", "chunk_003"]
  }
}
```

---

## LLM 配置

### 配置文件

所有工具共用 `aidoc.conf` 配置：

```ini
[llm]
api = ollama              # ollama 或 openai
model = qwen3:8b          # 模型名称
api_url = http://localhost:11434
api_key =                  # 仅 openai 需要
temperature = 0.3
timeout = 120
```

**搜索顺序：** `./aidoc.conf` → `~/.config/aidoc/aidoc.conf`

**优先级：** CLI 参数 > 配置文件 > 默认值

### CLI 参数覆盖

所有支持 LLM 的工具都接受以下参数：

```bash
--api ollama|openai    # 覆盖后端
--model MODEL          # 覆盖模型
--api-url URL          # 覆盖 API 地址
--api-key KEY          # 覆盖 API Key
--no-llm               # 禁用 LLM
```

### 推荐模型

| 场景 | Ollama 推荐 | OpenAI 推荐 |
|------|------------|-------------|
| 通用（中文文档） | `qwen3:8b` | `gpt-4o-mini` |
| 快速处理 | `llama3.2:3b` | `gpt-4o-mini` |
| 复杂文档 | `deepseek-r1:32b` | `gpt-4o` |

---

## 批量处理

```bash
# 批量转换 PDF
for f in docs/*.pdf; do
    python3 aidoc_convert.py "$f"
done

# 批量清理
for f in docs/*.md; do
    python3 aidoc_strip.py "$f" --no-llm
done

# 完整流水线
for f in docs/*.pdf; do
    base="${f%.pdf}"
    python3 aidoc_convert.py "$f"
    python3 aidoc_strip.py "${base}.md" --no-llm
    python3 aidoc_fix_hierarchy.py "${base}_clean.md" --no-llm
    python3 aidoc_fix_codeblocks.py "${base}_clean.md"
    python3 aidoc_index.py "${base}_clean.md" --no-llm
done
```

---

## 测试

### 安装测试依赖

```bash
pip3 install pytest
```

### 离线单元测试（无需外部服务）

```bash
# 84 个测试，< 1 秒
pytest tests/ -m "not integration" -v
```

覆盖所有核心逻辑：LLM 客户端工厂、JSON 提取、代码块检测、标题提取、
四层页眉检测、编号推断、内联小节识别、代码块分析修复、索引构建序列化。

### 集成测试（需要 Ollama + qwen3:8b）

```bash
# 7 个测试，约 3 分钟
pytest tests/ -m integration -v
```

验证 LLM 实际调用：Ollama 连接、LLM 辅助清理、LLM 辅助层级推断、
LLM 生成摘要关键字、strip → fix_hierarchy 端到端管线。

### 全量测试

```bash
pytest tests/ -v
```

无 Ollama 时集成测试自动跳过，不会报错。

---

## 文件结构

```
ai-doctool2/
├── README.md                   README
├── ARCHITECTURE.md             architecture doc (for AI maintenance)
├── aidoc.conf                  LLM config (Ollama default)
├── aidoc.openai.example.conf   OpenAI config example
├── requirements.txt            Python dependencies
├── .gitignore
│
│   shared modules
├── aidoc_llm.py                unified LLM client
├── aidoc_utils.py              common utilities
│
│   pipeline tools
├── aidoc_convert.py            [1] PDF -> Markdown
├── aidoc_strip.py              [2] strip headers & footers
├── aidoc_fix_hierarchy.py      [3] fix heading hierarchy
├── aidoc_fix_codeblocks.py     [4] fix code block boundaries
├── aidoc_index.py              [5] RAG semantic index
│
│   tests
└── tests/
    ├── conftest.py             fixtures + mock LLM
    ├── samples/                test sample markdowns
    ├── test_llm.py             LLM client tests
    ├── test_utils.py           utility tests
    ├── test_strip.py           strip tests
    ├── test_fix_hierarchy.py   hierarchy fixer tests
    ├── test_fix_codeblocks.py  codeblock fixer tests
    ├── test_index.py           indexer tests
    └── test_integration.py     integration tests (need Ollama)
```

---

## 依赖说明

| 依赖 | 用途 | 必需 |
|------|------|------|
| `requests` | HTTP 客户端（LLM API 调用） | 是 |
| `docling` | PDF 解析引擎 | 仅 aidoc_convert |
| `docling-core` | Docling 数据模型 | 仅 aidoc_convert |
| Ollama 服务 | 本地 LLM 推理 | 可选（--no-llm 跳过） |

---

## 设计原则

1. **通用化** — 不 hardcode 特定文档内容，使用通用结构特征
2. **多层置信度** — 统计 → 规则 → LLM，逐层收敛
3. **安全优先** — 宁可漏删不误删，代码块边界严格保护
4. **模块解耦** — 每个工具独立可用，LLM 后端可插拔
5. **配置统一** — 一个 `aidoc.conf` 管所有工具的 LLM 设置
