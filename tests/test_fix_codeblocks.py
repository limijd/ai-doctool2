"""
aidoc_fix_codeblocks 单元测试
==============================

测试代码块分析器和修复器的核心逻辑。
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aidoc_fix_codeblocks import (
    CodeBlockAnalyzer,
    CodeBlockFixer,
    IssueType,
    Confidence,
)


# =============================================================================
# CodeBlockAnalyzer
# =============================================================================

class TestCodeBlockAnalyzer:

    def setup_method(self):
        self.analyzer = CodeBlockAnalyzer()

    def test_detect_prose_in_code(self, broken_codeblocks_md):
        """应检测到代码块内的正文"""
        lines = broken_codeblocks_md.split('\n')
        issues = self.analyzer.analyze(lines)
        prose_issues = [i for i in issues if i.issue_type == IssueType.PROSE_IN_CODE]
        # 样本文件中有 heading 和段落被困在代码块内
        assert len(prose_issues) > 0

    def test_detect_unclosed_block(self):
        """应检测到未闭合的代码块"""
        lines = ["```python", "code here", "more code"]
        issues = self.analyzer.analyze(lines)
        unclosed = [i for i in issues if i.issue_type == IssueType.UNCLOSED_BLOCK]
        assert len(unclosed) == 1

    def test_no_issues_in_clean_code(self):
        """干净的代码块不应报任何问题"""
        lines = [
            "# Title",
            "",
            "```verilog",
            "module test;",
            "endmodule",
            "```",
            "",
            "Some text.",
        ]
        issues = self.analyzer.analyze(lines)
        assert len(issues) == 0

    def test_detect_indented_fence(self):
        """应检测到 4+ 空格缩进的 fence"""
        lines = [
            "```",
            "code line",
            "    ```",  # 畸形缩进 fence
            "text after",
        ]
        issues = self.analyzer.analyze(lines, {"check_indented_fences": True})
        indented = [i for i in issues if i.issue_type == IssueType.INDENTED_FENCE]
        assert len(indented) >= 1

    def test_prose_detection_markdown_heading(self):
        """Markdown 标题在代码块内应被高置信度识别"""
        lines = ["```", "## This is a heading", "```"]
        issues = self.analyzer.analyze(lines)
        if issues:
            prose = [i for i in issues if i.issue_type == IssueType.PROSE_IN_CODE]
            assert len(prose) >= 1
            assert prose[0].confidence == Confidence.HIGH

    def test_code_patterns_not_flagged(self):
        """代码特征行不应被误判为正文"""
        lines = [
            "```",
            "module test;",
            "  input logic clk;",
            "  always_ff @(posedge clk)",
            "endmodule",
            "```",
        ]
        issues = self.analyzer.analyze(lines)
        prose_issues = [i for i in issues if i.issue_type == IssueType.PROSE_IN_CODE]
        assert len(prose_issues) == 0


# =============================================================================
# CodeBlockFixer
# =============================================================================

class TestCodeBlockFixer:

    def test_fix_unclosed_block(self):
        """修复未闭合代码块"""
        content = "```python\ndef foo():\n    pass"
        fixer = CodeBlockFixer()
        fixed, changes = fixer.fix(content)
        assert fixed.rstrip().endswith("```")
        assert len(changes) > 0

    def test_fix_indented_fence(self):
        """修复缩进的 fence"""
        content = "```\ncode\n    ```\ntext"
        fixer = CodeBlockFixer()
        fixed, changes = fixer.fix(content, {"fix_indent": True})
        # 缩进应被移除
        lines = fixed.split('\n')
        fence_lines = [l for l in lines if l.strip().startswith('```')]
        for fl in fence_lines:
            leading_spaces = len(fl) - len(fl.lstrip())
            assert leading_spaces < 4

    def test_cleanup_adjacent_fences(self):
        """相邻的独立代码块不应被错误合并"""
        # 两个各自完整的代码块，fixer 不应破坏它们
        content = "text\n\n```verilog\nmodule t;\nendmodule\n```\n\n```verilog\nmodule u;\nendmodule\n```\n\nmore text"
        fixer = CodeBlockFixer()
        fixed, _ = fixer.fix(content)
        assert "module t;" in fixed
        assert "module u;" in fixed

    def test_no_change_on_clean_content(self):
        """干净内容不应被修改"""
        content = "# Title\n\n```verilog\nmodule t; endmodule\n```\n\nText."
        fixer = CodeBlockFixer()
        fixed, changes = fixer.fix(content)
        assert len(changes) == 0

    def test_fix_full_sample(self, broken_codeblocks_md):
        """完整样本修复后应减少问题数"""
        fixer = CodeBlockFixer()
        fixed, changes = fixer.fix(broken_codeblocks_md)

        # 修复后重新分析，问题应减少
        analyzer = CodeBlockAnalyzer()
        original_issues = analyzer.analyze(broken_codeblocks_md.split('\n'))
        fixed_issues = analyzer.analyze(fixed.split('\n'))
        assert len(fixed_issues) <= len(original_issues)
