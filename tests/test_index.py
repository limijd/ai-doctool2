"""
aidoc_index 单元测试
=====================

测试 Markdown 解析、切片、索引构建。
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aidoc_index import (
    MarkdownParser,
    IndexBuilder,
    ChunkInfo,
    TOCNode,
    DocumentIndex,
)


# =============================================================================
# MarkdownParser
# =============================================================================

class TestMarkdownParser:

    def test_load_file(self, samples_dir):
        """应正确加载文件"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        assert parser.file_size > 0
        assert len(parser.lines) > 0

    def test_get_headings(self, samples_dir):
        """应提取标题并跳过代码块"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        depth = parser.get_depth_level()
        headings = parser.get_headings(depth)
        assert len(headings) > 0
        # 所有标题都应是真标题，不含代码块内的
        titles = [h[2] for h in headings]
        assert "JTAG Boundary Scan Architecture" in titles

    def test_depth_auto_select(self, samples_dir):
        """小文件应选择细粒度切分"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        depth = parser.get_depth_level()
        # 样本文件很小，应选择深切分
        assert depth >= 3

    def test_get_chunk_content(self, samples_dir):
        """应能获取指定行范围的内容"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        content = parser.get_chunk_content(1, 3)
        assert len(content) > 0


# =============================================================================
# IndexBuilder (--no-llm 路径)
# =============================================================================

class TestIndexBuilder:

    def test_build_without_llm(self, samples_dir):
        """无 LLM 应生成纯结构索引"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=None)
        index = builder.build(use_llm=False)

        assert isinstance(index, DocumentIndex)
        assert len(index.chunks) > 0
        assert index.depth_level > 0
        assert index.total_lines > 0

    def test_chunks_have_metadata(self, samples_dir):
        """每个 chunk 应有完整的元数据"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=None)
        index = builder.build(use_llm=False)

        for chunk_id, chunk in index.chunks.items():
            assert chunk.id == chunk_id
            assert chunk.title
            assert chunk.level > 0
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert chunk.line_count > 0
            assert chunk.char_count > 0

    def test_chunks_no_summary_without_llm(self, samples_dir):
        """无 LLM 时 chunk 不应有摘要"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=None)
        index = builder.build(use_llm=False)

        for chunk in index.chunks.values():
            assert chunk.summary == ""
            assert chunk.keywords == []

    def test_toc_tree_structure(self, samples_dir):
        """TOC 树应有层次结构"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=None)
        index = builder.build(use_llm=False)

        assert index.toc_tree  # 非空
        assert "children" in index.toc_tree

    def test_serialization(self, samples_dir):
        """索引应能序列化为 JSON"""
        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=None)
        index = builder.build(use_llm=False)

        json_str = json.dumps(index.to_dict(), ensure_ascii=False, indent=2)
        parsed = json.loads(json_str)
        assert "chunks" in parsed
        assert "toc_tree" in parsed
        assert "keyword_index" in parsed


# =============================================================================
# 数据结构
# =============================================================================

class TestDataStructures:

    def test_toc_node_to_dict(self):
        root = TOCNode(id="root", title="Root", level=0)
        child = TOCNode(id="c1", title="Child", level=1)
        root.children.append(child)

        d = root.to_dict()
        assert d["id"] == "root"
        assert len(d["children"]) == 1
        assert d["children"][0]["title"] == "Child"
