# ARCHITECTURE — aidoc 工具链架构文档

> 本文档面向 AI 辅助维护场景。当你需要修改或扩展此工具链时，先阅读本文档。

---

## 整体架构

```
                    aidoc.conf (LLM 配置)
                         │
                    aidoc_llm.py (统一 LLM 客户端)
                    ├── OllamaClient
                    └── OpenAIClient
                         │
                    aidoc_utils.py (共享工具)
                    ├── 代码块检测
                    ├── 标题提取
                    └── CLI 辅助
                         │
        ┌────────┬───────┼───────┬──────────┐
        ▼        ▼       ▼       ▼          ▼
   convert    clean    fix_     fix_      index
   (PDF→MD)  (页眉    hierarchy codeblocks (RAG
              页脚)   (标题)   (代码块)    索引)
```

### 依赖关系

- `aidoc_convert.py` — 独立，不依赖 aidoc_llm（Docling 自包含）
- `aidoc_strip.py` — 依赖 aidoc_llm + aidoc_utils
- `aidoc_fix_hierarchy.py` — 依赖 aidoc_llm + aidoc_utils
- `aidoc_fix_codeblocks.py` — 依赖 aidoc_llm + aidoc_utils
- `aidoc_index.py` — 依赖 aidoc_llm + aidoc_utils

### 数据流

```
PDF → [convert] → raw.md → [clean] → clean.md → [fix_hierarchy] → fixed.md
                                                                      │
                                                    [fix_codeblocks] ─┘
                                                          │
                                                    [index] → .index.json
```

每一步的输出都是标准 Markdown 文件，可以单独运行任何工具。

---

## 模块详解

### aidoc_llm.py — LLM 客户端

**职责**：为所有工具提供统一的 LLM 调用接口。

**关键设计**：

1. **三级配置优先级**：CLI 参数 > aidoc.conf > 默认值
2. **双后端支持**：Ollama（本地） + OpenAI（云端/兼容服务）
3. **统一接口**：`generate(prompt, system, temperature, max_tokens) → str`

**核心类**：

| 类 | 说明 |
|---|---|
| `LLMClient` | 抽象基类，定义 generate() 接口 |
| `OllamaClient(LLMClient)` | Ollama REST API 实现 |
| `OpenAIClient(LLMClient)` | OpenAI Chat Completions 实现 |

**工厂函数**：

| 函数 | 说明 |
|---|---|
| `load_config()` | 加载 aidoc.conf |
| `add_llm_args(parser)` | 添加 CLI LLM 参数组 |
| `create_llm_client(args)` | 从 argparse 结果创建客户端 |
| `extract_json(text)` | 从 LLM 响应提取 JSON |

**修改指南**：
- 添加新后端：继承 `LLMClient`，实现 `generate()` 和 `check_connection()`
- 修改配置格式：改 `load_config()` 和 `CONFIG_SEARCH_PATHS`
- 所有工具通过 `create_llm_client(args)` 获取客户端，无需直接构造

---

### aidoc_utils.py — 共享工具

**职责**：跨工具复用的 Markdown 解析和 CLI 辅助功能。

**关键常量**：

| 常量 | 说明 |
|---|---|
| `HEADING_PATTERN` | Markdown 标题正则 `^#{1,6}\s+(.+)$` |
| `CODE_FENCE_PATTERN` | 合法代码块栅栏（0-3 空格缩进） |
| `INDENTED_FENCE_PATTERN` | 畸形栅栏（4+ 空格缩进） |
| `SIMPLE_FENCE_PATTERN` | 简单 ``` 检测 |

**关键函数**：

| 函数 | 说明 |
|---|---|
| `find_code_block_lines(lines)` | 返回代码块内行号集合（0-based） |
| `extract_headings(lines, max_level)` | 提取标题列表（自动跳过代码块） |
| `normalize_whitespace(text, max_empty)` | 压缩连续空行 |
| `print_banner(title)` | 打印标题横幅 |
| `print_stats(stats)` | 打印统计表格 |

**修改指南**：
- 添加新的 Markdown 解析功能放这里
- 新工具需要的公共 CLI 辅助也放这里
- 保持零外部依赖（只用标准库）

---

### aidoc_convert.py — PDF 转 Markdown

**职责**：PDF → Markdown 转换，管线第一步。

**核心类**：`DoclingPdfConverter`

**不使用 LLM**，是唯一不依赖 aidoc_llm 的工具。

**关键特性**：
- 代码块识别增强（Docling enrich-code）
- 公式 LaTeX 转换
- 高精度表格提取
- 页眉页脚初步过滤（基于 Docling 布局模型 + 位置统计）
- 跨页代码块合并
- OCR 支持

**修改指南**：
- Docling 配置在 `_create_converter()` 方法中
- 页眉页脚检测在 `detect_headers_footers_by_position()` 函数中
- 代码块合并在 `merge_consecutive_code_blocks()` 函数中

---

### aidoc_strip.py — 页眉页脚剥离

**职责**：智能清理 PDF 转换残留的页眉、页脚、页码、水印。

**四层架构**：

```
Layer 1: PatternDetector    统计模式检测
         │                  - 精确频率分析
         │                  - 归一化频率（忽略页码变化）
         │                  - 相似度聚类（OCR 容错）
         ▼
Layer 2: HeuristicFilter    启发式分类
         │                  - 页码 (+0.25)、页眉 (+0.15)
         │                  - 页脚 (+0.15)、水印 (+0.20)
         ▼
Layer 3: LLM 验证           仅 [0.35, 0.6) 区间触发
         │
         ▼
Layer 3.5: CodeBlockCleaner 代码块内清理
         │                  - 特征词组匹配
         │                  - 代码语法保护
         ▼
Layer 4: ContentMerger      内容合并
                            - 代码块合并
                            - 表格合并
                            - 空行压缩
```

**置信度系统**：
- 初始值 = `0.4 + (count/total)*50 + (count/100)*0.3`
- 各层叠加调整
- `≥0.6` 直接移除，`[0.35, 0.6)` 触发 LLM，`<0.35` 丢弃

**主处理类**：`MarkdownCleaner`，协调全部层级。

**修改指南**：
- 添加新的检测规则：改 `HeuristicFilter` 的分类模式
- 调整置信度阈值：改 `MarkdownCleaner.HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE`
- 添加新的页脚模式：加到 `PatternDetector._detect_common_footers()`
- 代码块内清理：改 `CodeBlockCleaner.KEYWORD_GROUPS`

---

### aidoc_fix_hierarchy.py — 标题层级修复

**职责**：修复 PDF 转换后标题层级退化（如全部变成 `##`）。

**五阶段算法**：

```
Phase 0: 页眉污染检测
         检测重复出现的文档标题，保留首次，标记后续为污染

Phase 1: 规则骨架构建 (最高置信度)
         编号模式 → 层级映射:
           1.       → H2 (99.5%)
           1.1      → H3 (99.5%)
           1.1.1    → H4 (99.5%)
           A. / Annex A → H2 (90%)

Phase 2: 区间归属分析
         无编号标题 → 属于 [前编号, 后编号) 区间
         层级约束: 必须 > 区间父级

Phase 3: 内联小节标题
         Rules, Permissions, Example 等 → 父级 + 1

Phase 4: LLM 辅助推断
         剩余无法确定的标题，带约束信息调用 LLM
         约束: 已知标题上下文 + 区间范围
```

**核心类**：

| 类 | 说明 |
|---|---|
| `HeadingAnalyzer` | 标题提取、编号推断、内联小节检测 |
| `HierarchyFixer` | 五阶段修复主逻辑 |
| `MarkdownWriter` | 将修复结果写回 Markdown |

**数据结构**：`HeadingInfo` dataclass 包含行号、原始/推断层级、编号、方法、置信度。

**修改指南**：
- 添加新的编号模式：改 `PATTERNS` 字典和 `infer_level_from_numbering()`
- 添加内联小节关键字：改 `INLINE_SECTION_TITLES` 集合
- 调整 LLM prompt：改 `_fix_with_llm()` 中的 prompt 模板
- 页眉检测模式：改 `HEADER_FOOTER_PATTERNS` 列表

---

### aidoc_fix_codeblocks.py — 代码块边界修复

**职责**：修复 ``` 标记错位导致的代码/正文混淆。

**问题分类**：

| 类型 | 说明 | 置信度 |
|---|---|---|
| `PROSE_IN_CODE` | 代码块内出现 Markdown 标题/段落 | HIGH-LOW |
| `INDENTED_FENCE` | 4+ 空格缩进的 ``` | HIGH-LOW |
| `UNCLOSED_BLOCK` | 缺少闭合 ``` | HIGH |

**检测策略**：
- 正文特征：Markdown 标题、长段落（>50 字符）、表格
- 代码特征：HDL/C/Python 关键字、语法符号（`;`, `:=`, `<=`, `=>`）
- 排除规则：注释、字符串字面量不算正文

**修复流程**：
1. 修复缩进栅栏（去前导空格）
2. 正文区域前后插入 ```
3. 闭合未关闭的代码块
4. 清理相邻的空代码块

**修改指南**：
- 添加代码语言识别：改 `CODE_PATTERNS` 列表
- 添加正文识别：改 `PROSE_PATTERNS` 列表
- 调整置信度评估：改 `_assess_prose_confidence()` / `_assess_indented_fence_confidence()`

---

### aidoc_index.py — RAG 语义索引

**职责**：将 Markdown 按标题层级切片，生成 JSON 索引供 RAG 检索。

**自动深度策略**：
- 小文件 (<50KB) → 切到 H4
- 中文件 (50-200KB) → 切到 H3
- 大文件 (>200KB) → 切到 H2

**索引结构**：

```json
{
  "source_file": "...",
  "toc_tree": { /* 层次化目录 */ },
  "chunks": {
    "chunk_001": {
      "title": "...",
      "level": 2,
      "start_line": 10,
      "end_line": 150,
      "summary": "LLM 生成的摘要",
      "keywords": ["关键字"]
    }
  },
  "keyword_index": {
    "keyword": ["chunk_001", "chunk_003"]
  }
}
```

**核心类**：

| 类 | 说明 |
|---|---|
| `MarkdownParser` | 文件加载、标题提取、内容切取 |
| `IndexBuilder` | 切片创建、摘要生成、索引构建 |
| `ChunkInfo` | 切片元数据 |
| `TOCNode` | 目录树节点 |
| `DocumentIndex` | 完整索引数据 |

**修改指南**：
- 调整切分策略：改 `DEPTH_CONFIG` 和文件大小阈值
- 修改摘要 prompt：改 `_summarize_chunk()` 中的 prompt
- 扩展索引字段：改 `ChunkInfo` dataclass 和 `DocumentIndex.to_dict()`

---

## 扩展指南

### 添加新的管线工具

1. 创建 `aidoc_新工具.py`
2. 如需 LLM：`from aidoc_llm import add_llm_args, create_llm_client`
3. 如需 Markdown 解析：`from aidoc_utils import find_code_block_lines, extract_headings`
4. CLI 模式：`argparse` + `add_llm_args(parser)` + `create_llm_client(args)`
5. 更新 README.md 和本文档

### 添加新的 LLM 后端

1. 在 `aidoc_llm.py` 中创建新类，继承 `LLMClient`
2. 实现 `generate()`、`check_connection()`、`backend_name`
3. 在 `create_llm_client_from_config()` 中添加 `elif api == "新后端"` 分支
4. 在 `add_llm_args()` 中更新 `choices`

### 配置系统

```
aidoc.conf (INI 格式)
  └── [llm] 节
       ├── api          后端类型
       ├── model        模型名
       ├── api_url      API 地址
       ├── api_key      API Key
       ├── temperature  温度
       └── timeout      超时
```

搜索路径：`./aidoc.conf` → `~/.config/aidoc/aidoc.conf`

---

## 代码规范

- 注释语言：中文
- 类型标注：使用 Python 3.9+ 语法（`list[str]` 而非 `List[str]`）
- 命名：`aidoc_` 前缀，`snake_case`
- 每个工具都是独立脚本，`python3 aidoc_xxx.py` 直接运行
- 共享代码通过同目录 import 复用
