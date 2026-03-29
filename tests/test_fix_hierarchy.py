"""
aidoc_fix_hierarchy 单元测试
============================

测试编号推断、区间归属、内联小节等核心算法。
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aidoc_fix_hierarchy import (
    HeadingAnalyzer,
    HierarchyFixer,
    MarkdownWriter,
    HeadingInfo,
    PATTERNS,
)


# =============================================================================
# 编号推断
# =============================================================================

class TestNumberingInference:
    """测试 infer_level_from_numbering 的编号识别"""

    def test_level1(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("1. Overview")
        assert level == 2
        assert numbering == "1."
        assert conf > 0.9

    def test_level2(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("1.1 Scope")
        assert level == 3
        assert conf > 0.9

    def test_level3(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("1.1.1 Details")
        assert level == 4
        assert conf > 0.9

    def test_level4(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("1.1.1.1 Sub-details")
        assert level == 5
        assert conf > 0.9

    def test_annex(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("A. Formal syntax")
        assert level == 2
        assert conf > 0.8

    def test_annex_sub(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("A.1 Source text")
        assert level == 3

    def test_annex_sub2(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("A.1.1 Library")
        assert level == 4

    def test_no_numbering(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("Abstract")
        assert level == -1
        assert conf == 0.0

    def test_figure(self):
        level, numbering, conf = HeadingAnalyzer.infer_level_from_numbering("Figure 3")
        assert level == -1  # 需要上下文判断


# =============================================================================
# 内联小节标题检测
# =============================================================================

class TestInlineSectionDetection:

    def test_rules(self):
        assert HeadingAnalyzer.is_inline_section_title("Rules") is True

    def test_permissions(self):
        assert HeadingAnalyzer.is_inline_section_title("Permissions") is True

    def test_example_numbered(self):
        assert HeadingAnalyzer.is_inline_section_title("Example 1") is True

    def test_notes_with_colon(self):
        assert HeadingAnalyzer.is_inline_section_title("Notes:") is True

    def test_regular_title(self):
        assert HeadingAnalyzer.is_inline_section_title("Module declarations") is False

    def test_overview(self):
        assert HeadingAnalyzer.is_inline_section_title("Overview") is True


# =============================================================================
# 完整修复流程 (--no-llm)
# =============================================================================

class TestHierarchyFixer:

    def test_fix_flat_hierarchy(self, flat_hierarchy_md):
        """全部 ## 的文档应被修复为正确层级"""
        lines = flat_hierarchy_md.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=None)
        result = fixer.fix(use_llm=False, verbose=False)

        # 应该有修复发生
        assert result.fixed_headings > 0
        assert result.rule_based_fixes > 0

        # 检查具体层级：1. -> H2, 1.1 -> H3, 1.1.1 -> H4
        heading_map = {h.title: h for h in result.headings if not h.is_header_pollution}

        h_overview = heading_map.get("1. Overview")
        if h_overview:
            assert h_overview.inferred_level == 2

        h_scope = heading_map.get("1.1 Scope")
        if h_scope:
            assert h_scope.inferred_level == 3

        h_features = heading_map.get("1.1.1 Language features")
        if h_features:
            assert h_features.inferred_level == 4

    def test_annex_levels(self, flat_hierarchy_md):
        """附录 A./A.1/A.1.1 应被正确识别"""
        lines = flat_hierarchy_md.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=None)
        result = fixer.fix(use_llm=False, verbose=False)

        heading_map = {h.title: h for h in result.headings}

        h_annex_a = heading_map.get("A. Formal syntax")
        if h_annex_a:
            assert h_annex_a.inferred_level == 2

        h_annex_a1 = heading_map.get("A.1 Source text")
        if h_annex_a1:
            assert h_annex_a1.inferred_level == 3

    def test_inline_sections(self, flat_hierarchy_md):
        """Rules/Permissions/Notes 等应被识别为内联小节"""
        lines = flat_hierarchy_md.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=None)
        result = fixer.fix(use_llm=False, verbose=False)

        inline_titles = [h.title for h in result.headings if h.is_inline_section]
        assert "Rules" in inline_titles or "Permissions" in inline_titles

    def test_writer_output(self, flat_hierarchy_md):
        """MarkdownWriter 应正确输出修复后的 Markdown"""
        lines = flat_hierarchy_md.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=None)
        result = fixer.fix(use_llm=False, verbose=False)

        writer = MarkdownWriter(lines, result.headings)
        output = writer.generate()

        # 输出应包含不同层级的标题
        assert "## " in output  # H2
        assert "### " in output  # H3
        assert "#### " in output  # H4
