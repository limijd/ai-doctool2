"""
aidoc_utils 单元测试
=====================

测试代码块检测、标题提取、文本处理等共享功能。
"""

import pytest
from aidoc_utils import (
    find_code_block_lines,
    extract_headings,
    normalize_whitespace,
    truncate,
    HEADING_PATTERN,
    CODE_FENCE_PATTERN,
)


# =============================================================================
# find_code_block_lines
# =============================================================================

class TestFindCodeBlockLines:

    def test_no_code_blocks(self):
        lines = ["# Title", "Some text", "More text"]
        assert find_code_block_lines(lines) == set()

    def test_single_code_block(self):
        lines = ["text", "```", "code line", "```", "text"]
        result = find_code_block_lines(lines)
        assert 1 in result  # 开始 fence
        assert 2 in result  # 代码行
        assert 3 in result  # 结束 fence
        assert 0 not in result
        assert 4 not in result

    def test_multiple_code_blocks(self):
        lines = ["```", "a", "```", "gap", "```", "b", "```"]
        result = find_code_block_lines(lines)
        assert 3 not in result  # gap 不在代码块内
        assert 0 in result
        assert 1 in result
        assert 4 in result
        assert 5 in result

    def test_code_block_with_language(self):
        lines = ["```python", "print('hi')", "```"]
        result = find_code_block_lines(lines)
        assert result == {0, 1, 2}

    def test_unclosed_code_block(self):
        """未闭合的代码块，后续所有行都算在内"""
        lines = ["text", "```", "code", "still code"]
        result = find_code_block_lines(lines)
        assert 0 not in result
        assert 1 in result
        assert 2 in result
        assert 3 in result


# =============================================================================
# extract_headings
# =============================================================================

class TestExtractHeadings:

    def test_basic_headings(self):
        lines = ["# H1", "text", "## H2", "### H3"]
        headings = extract_headings(lines)
        assert len(headings) == 3
        assert headings[0] == (1, 1, "H1")
        assert headings[1] == (3, 2, "H2")
        assert headings[2] == (4, 3, "H3")

    def test_skip_code_block_headings(self):
        """代码块内的 # 不应被识别为标题"""
        lines = ["# Real", "```", "# Fake", "```", "## Real2"]
        headings = extract_headings(lines)
        assert len(headings) == 2
        assert headings[0][2] == "Real"
        assert headings[1][2] == "Real2"

    def test_max_level_filter(self):
        lines = ["# H1", "## H2", "### H3", "#### H4"]
        headings = extract_headings(lines, max_level=2)
        assert len(headings) == 2

    def test_empty_input(self):
        assert extract_headings([]) == []


# =============================================================================
# normalize_whitespace
# =============================================================================

class TestNormalizeWhitespace:

    def test_compress_empty_lines(self):
        text = "a\n\n\n\n\nb"
        result = normalize_whitespace(text, max_empty=2)
        assert result.count('\n') == 3  # a + 2 空行 + b = 3 个换行

    def test_no_change_needed(self):
        text = "a\n\nb"
        assert normalize_whitespace(text) == text

    def test_custom_max(self):
        text = "a\n\n\nb"
        result = normalize_whitespace(text, max_empty=1)
        assert result == "a\n\nb"


# =============================================================================
# truncate
# =============================================================================

class TestTruncate:

    def test_short_string(self):
        assert truncate("hello", 10) == "hello"

    def test_long_string(self):
        result = truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_exact_length(self):
        assert truncate("12345", 5) == "12345"


# =============================================================================
# 正则常量
# =============================================================================

class TestPatterns:

    def test_heading_pattern(self):
        assert HEADING_PATTERN.match("# Title")
        assert HEADING_PATTERN.match("### Deep heading")
        assert not HEADING_PATTERN.match("Not a heading")
        assert not HEADING_PATTERN.match("#No space")

    def test_code_fence_pattern(self):
        assert CODE_FENCE_PATTERN.match("```")
        assert CODE_FENCE_PATTERN.match("```python")
        assert CODE_FENCE_PATTERN.match("   ```")  # 0-3 空格
        assert not CODE_FENCE_PATTERN.match("    ```")  # 4 空格 = 不合法
