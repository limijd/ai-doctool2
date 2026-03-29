#!/usr/bin/env python3
"""
aidoc_index.py - RAG 语义索引工具
==================================

对 Markdown 文件进行语义切片与索引构建，通过 LLM 生成章节摘要和关键字。

功能：
  1. 基于标题层级的语义切片（自动根据文件大小选择切分粒度）
  2. LLM 驱动的章节摘要生成与关键字提取
  3. 层级目录树（TOC）构建
  4. 倒排关键字索引
  5. 结构化 JSON 索引输出
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from aidoc_llm import LLMClient, add_llm_args, create_llm_client, extract_json
from aidoc_utils import (
    extract_headings,
    find_code_block_lines,
    print_banner,
    print_stats,
    ProgressPrinter,
)


# =============================================================================
# 切片策略配置
# =============================================================================
#
# 文件大小 → 切分深度映射：
#   - 小文件（<50KB）：文档结构简单，切到 H4 获取细粒度语义单元
#   - 中文件（50-200KB）：平衡精度与数量，切到 H3
#   - 大文件（>200KB）：避免 chunk 过多导致索引膨胀，只切到 H2
#
# 这一策略确保无论文档大小，每个 chunk 都保持合理的上下文长度，
# 既不会太短（丢失语义）也不会太长（降低检索精度）。

SMALL_FILE_THRESHOLD = 50 * 1024       # 50KB
MEDIUM_FILE_THRESHOLD = 200 * 1024     # 200KB

DEPTH_CONFIG = {
    "small": 4,     # 小文件：切到 H4
    "medium": 3,    # 中文件：切到 H3
    "large": 2,     # 大文件：切到 H2
}


# =============================================================================
# 数据结构
# =============================================================================
#
# 索引由三层结构组成：
#   ChunkInfo   - 单个语义切片的完整信息（位置、内容摘要、关键字）
#   TOCNode     - 目录树节点，反映文档的层级结构
#   DocumentIndex - 顶层索引，聚合所有 chunk、TOC 和倒排索引

@dataclass
class ChunkInfo:
    """语义切片信息"""
    id: str                                # 唯一标识（chunk_001 格式）
    title: str                             # 章节标题
    level: int                             # 标题层级（1-6）
    start_line: int                        # 起始行号（1-based）
    end_line: int                          # 结束行号（含）
    line_count: int                        # 行数
    char_count: int                        # 字符数
    content_preview: str                   # 内容预览（前 200 字符）
    summary: str = ""                      # LLM 生成的摘要
    keywords: list = field(default_factory=list)   # LLM 提取的关键字
    children: list = field(default_factory=list)   # 子章节 ID 列表


@dataclass
class TOCNode:
    """目录树节点"""
    id: str
    title: str
    level: int
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DocumentIndex:
    """文档索引顶层结构"""
    source_file: str                       # 源文件路径
    file_size: int                         # 文件大小（字节）
    total_lines: int                       # 总行数
    depth_level: int                       # 实际使用的切分深度
    toc_tree: dict                         # TOC 树（序列化后）
    chunks: dict                           # chunk_id -> ChunkInfo
    keyword_index: dict                    # keyword -> [chunk_id, ...]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "file_size": self.file_size,
            "total_lines": self.total_lines,
            "depth_level": self.depth_level,
            "toc_tree": self.toc_tree,
            "chunks": {k: asdict(v) for k, v in self.chunks.items()},
            "keyword_index": self.keyword_index,
            "metadata": self.metadata,
        }


# =============================================================================
# Markdown 解析器
# =============================================================================

class MarkdownParser:
    """
    Markdown 文件解析器。

    负责文件加载、标题提取和内容切片。标题提取和代码块检测
    委托给 aidoc_utils 中的共享实现，确保工具链行为一致。
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.lines: list[str] = []
        self.file_size = 0
        self._load_file()

    def _load_file(self):
        """加载文件内容，记录文件大小"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"文件不存在: {self.filepath}")

        self.file_size = self.filepath.stat().st_size
        with open(self.filepath, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def get_depth_level(self) -> int:
        """根据文件大小自动确定切分深度"""
        if self.file_size < SMALL_FILE_THRESHOLD:
            return DEPTH_CONFIG["small"]
        elif self.file_size < MEDIUM_FILE_THRESHOLD:
            return DEPTH_CONFIG["medium"]
        else:
            return DEPTH_CONFIG["large"]

    def get_headings(self, max_level: int) -> list[tuple[int, int, str]]:
        """
        提取标题列表（自动跳过代码块中的伪标题）。

        委托 aidoc_utils.extract_headings() 实现，保证与其他工具的解析行为一致。

        Returns:
            [(行号(1-based), 层级, 标题文本), ...]
        """
        return extract_headings(self.lines, max_level)

    def get_chunk_content(self, start_line: int, end_line: int) -> str:
        """获取指定行范围的内容（行号均为 1-based）"""
        start_idx = start_line - 1
        end_idx = end_line
        return "".join(self.lines[start_idx:end_idx])


# =============================================================================
# 摘要生成
# =============================================================================

def summarize_chunk(llm: LLMClient, content: str, title: str) -> tuple[str, list[str]]:
    """
    调用 LLM 为单个 chunk 生成摘要和关键字。

    Args:
        llm:     LLM 客户端实例
        content: chunk 的原始文本
        title:   chunk 的标题

    Returns:
        (摘要文本, 关键字列表)；内容过短或调用失败时返回 ("", [])
    """
    # 内容过短，无需摘要
    if len(content.strip()) < 50:
        return "", []

    # 截断过长内容，避免超出 LLM 上下文窗口
    max_content = 4000
    if len(content) > max_content:
        content = content[:max_content] + "..."

    prompt = f"""请分析以下章节内容，生成简洁的摘要和关键字。

章节标题: {title}

章节内容:
{content}

请按以下 JSON 格式输出（只输出JSON，不要其他内容）:
{{
    "summary": "一句话摘要（50字以内）",
    "keywords": ["关键字1", "关键字2", "关键字3"]
}}
"""
    system = "你是一个文档分析助手，擅长提取文档的核心内容和关键信息。只输出JSON格式，不要输出其他解释。"

    response = llm.generate(prompt, system, temperature=0.2)

    # 使用共享的 JSON 提取函数解析响应
    data = extract_json(response)
    if data:
        summary = data.get("summary", "")[:100]
        keywords = data.get("keywords", [])[:5]
        return summary, keywords

    return "", []


# =============================================================================
# 索引构建器
# =============================================================================

class IndexBuilder:
    """
    文档索引构建器。

    构建流程：
      1. 提取标题 → 确定 chunk 边界
      2. 创建 ChunkInfo 列表
      3. （可选）调用 LLM 生成摘要和关键字
      4. 构建层级目录树
      5. 构建倒排关键字索引
      6. 组装 DocumentIndex
    """

    def __init__(self, parser: MarkdownParser, llm: Optional[LLMClient] = None):
        self.parser = parser
        self.llm = llm
        self.depth_level = parser.get_depth_level()
        self.chunks: dict[str, ChunkInfo] = {}
        self.keyword_index: dict[str, list[str]] = {}

    def build(self, use_llm: bool = True) -> DocumentIndex:
        """构建完整索引并返回 DocumentIndex"""
        print(f"文件大小: {self.parser.file_size / 1024:.1f} KB")
        print(f"切分深度: H1-H{self.depth_level}")
        print(f"总行数: {len(self.parser.lines)}")

        # 第一步：提取标题，确定 chunk 边界
        headings = self.parser.get_headings(self.depth_level)
        print(f"识别到 {len(headings)} 个章节")

        # 第二步：创建 chunk 数据
        self._create_chunks(headings)

        # 第三步：LLM 摘要生成
        if use_llm and self.llm:
            self._generate_summaries()

        # 第四步：构建目录树
        toc_tree = self._build_toc_tree(headings)

        # 第五步：构建倒排关键字索引
        self._build_keyword_index()

        # 组装最终索引
        return DocumentIndex(
            source_file=str(self.parser.filepath),
            file_size=self.parser.file_size,
            total_lines=len(self.parser.lines),
            depth_level=self.depth_level,
            toc_tree=toc_tree.to_dict() if toc_tree else {},
            chunks=self.chunks,
            keyword_index=self.keyword_index,
            metadata={
                "model": self.llm.model if self.llm else None,
                "depth_config": DEPTH_CONFIG,
            },
        )

    def _create_chunks(self, headings: list[tuple[int, int, str]]):
        """
        根据标题列表创建 chunk。

        每个 chunk 的范围是从当前标题行到下一个标题行之前（或文件末尾）。
        """
        total_lines = len(self.parser.lines)

        for i, (line_num, level, title) in enumerate(headings):
            # chunk 结束于下一个标题行之前，或文件末尾
            if i + 1 < len(headings):
                end_line = headings[i + 1][0] - 1
            else:
                end_line = total_lines

            chunk_id = f"chunk_{i + 1:03d}"
            content = self.parser.get_chunk_content(line_num, end_line)
            line_count = end_line - line_num + 1

            # 生成内容预览（压缩换行，截断到 200 字符）
            preview = content[:200].replace("\n", " ").strip()
            if len(content) > 200:
                preview += "..."

            self.chunks[chunk_id] = ChunkInfo(
                id=chunk_id,
                title=title,
                level=level,
                start_line=line_num,
                end_line=end_line,
                line_count=line_count,
                char_count=len(content),
                content_preview=preview,
            )

    def _generate_summaries(self):
        """使用 LLM 为每个 chunk 生成摘要和关键字"""
        progress = ProgressPrinter(total=len(self.chunks), prefix="摘要生成")

        for i, (chunk_id, chunk) in enumerate(self.chunks.items(), 1):
            progress.update(i, detail=chunk.title)

            content = self.parser.get_chunk_content(chunk.start_line, chunk.end_line)
            summary, keywords = summarize_chunk(self.llm, content, chunk.title)

            chunk.summary = summary
            chunk.keywords = keywords
            progress.item_done(success=bool(summary))

        progress.finish()

    def _build_toc_tree(self, headings: list[tuple[int, int, str]]) -> Optional[TOCNode]:
        """
        从扁平标题列表构建层级目录树。

        算法：使用栈维护当前路径上的祖先节点。
        遇到新标题时，回退栈直到找到层级更高（数字更小）的父节点，
        然后将新节点挂载为其子节点。
        """
        if not headings:
            return None

        # 虚拟根节点（level=0），作为所有顶层标题的父节点
        root = TOCNode(id="root", title="", level=0)
        stack = [root]
        chunk_idx = 0

        for line_num, level, title in headings:
            chunk_id = f"chunk_{chunk_idx + 1:03d}"
            node = TOCNode(id=chunk_id, title=title, level=level)

            # 回退栈：找到层级严格更高的祖先作为父节点
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)

            stack.append(node)
            chunk_idx += 1

        return root

    def _build_keyword_index(self):
        """
        构建倒排关键字索引（keyword -> chunk_id 列表）。

        关键字统一转小写以支持大小写无关检索。
        """
        for chunk_id, chunk in self.chunks.items():
            for keyword in chunk.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                if chunk_id not in self.keyword_index[keyword_lower]:
                    self.keyword_index[keyword_lower].append(chunk_id)


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Markdown 语义切片与 RAG 索引工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用示例:
  # 基本用法 - 使用默认模型处理文档
  %(prog)s document.md

  # 快速预览 - 只生成结构索引，不调用 LLM
  %(prog)s document.md --no-llm

  # 指定输出文件名
  %(prog)s document.md -o my_index.json

  # 使用其他模型
  %(prog)s document.md --model deepseek-r1:32b

  # 强制指定切分深度（覆盖自动检测）
  %(prog)s large_doc.md --depth 2   # 只切到 H2
  %(prog)s small_doc.md --depth 4   # 切到 H4

  # 批量处理
  for f in docs/*.md; do %(prog)s "$f"; done

切分策略 (自动根据文件大小选择):
  - 小文件 (<50KB):   细粒度，切到 H4
  - 中文件 (50-200KB): 中等粒度，切到 H3
  - 大文件 (>200KB):  粗粒度，切到 H2
        """,
    )

    parser.add_argument("input", help="输入的 Markdown 文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出的索引文件路径（默认: <input>.index_full.json 或 <input>.index.json）",
    )
    parser.add_argument(
        "--depth", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="强制指定切分深度（覆盖自动检测）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # 添加统一的 LLM 参数（--api, --model, --api-url, --api-key, --no-llm）
    add_llm_args(parser)

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = ".index.json" if getattr(args, "no_llm", False) else ".index_full.json"
        output_path = input_path.parent / f"{input_path.name}{suffix}"

    # 打印横幅
    use_llm = not getattr(args, "no_llm", False)
    model_display = getattr(args, "model", None) or "(自动)"
    print_banner("aidoc_index - RAG 语义索引工具")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"模型: {model_display if use_llm else '(不使用)'}")
    print()

    # 解析 Markdown
    md_parser = MarkdownParser(str(input_path))

    # 如果指定了深度，覆盖自动检测
    if args.depth:
        forced_depth = args.depth
        md_parser.get_depth_level = lambda: forced_depth

    # 创建 LLM 客户端（通过统一工厂函数）
    llm = create_llm_client(args) if use_llm else None

    # 构建索引
    print("开始构建索引...")
    builder = IndexBuilder(md_parser, llm)
    index = builder.build(use_llm=use_llm)

    # 保存索引
    print(f"\n保存索引到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, ensure_ascii=False, indent=2)

    # 输出统计
    print_stats(
        {
            "章节数量": len(index.chunks),
            "关键字数量": len(index.keyword_index),
            "切分深度": f"H1-H{index.depth_level}",
        },
        title="索引统计",
    )
    print("完成!")


if __name__ == "__main__":
    main()
