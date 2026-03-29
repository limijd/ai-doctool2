"""
aidoc_strip 单元测试
=====================

测试四层检测架构的各个组件。
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aidoc_strip import (
    PatternDetector,
    HeuristicFilter,
    ContentMerger,
    CodeBlockCleaner,
    MarkdownCleaner,
    DetectedPattern,
)


# =============================================================================
# Layer 1: PatternDetector
# =============================================================================

class TestPatternDetector:

    def test_detect_repeated_lines(self, messy_headers_md):
        """高频重复行应被检测为候选模式"""
        detector = PatternDetector(min_frequency=3)
        lines = messy_headers_md.split('\n')
        patterns = detector.analyze(lines)
        # "Copyright © 2023 IEEE. All rights reserved." 出现多次
        texts = [p.text for p in patterns]
        assert any("Copyright" in t for t in texts)

    def test_no_false_positives_on_unique_lines(self):
        """短而完全不同的行不应被精确频率或相似度检测到"""
        # 每行长度 < 10 且互不相似，跳过相似度聚类的候选阈值
        import random
        random.seed(42)
        lines = [f"{random.randint(100000, 999999)}" for _ in range(50)]
        detector = PatternDetector(min_frequency=5)
        patterns = detector.analyze(lines)
        assert len(patterns) == 0

    def test_normalized_frequency(self):
        """带页码变化的行应归一化后识别"""
        lines = []
        for i in range(20):
            lines.append(f"Page {i+1} of 20")
            lines.append(f"Content line {i}")
        detector = PatternDetector(min_frequency=5)
        patterns = detector.analyze(lines)
        # "Page N of 20" 系列应被归一化检测到
        assert len(patterns) > 0


# =============================================================================
# Layer 2: HeuristicFilter
# =============================================================================

class TestHeuristicFilter:

    def setup_method(self):
        self.filter = HeuristicFilter()

    def test_classify_page_number(self):
        """纯数字应分类为页码"""
        p = DetectedPattern(text="42", pattern_type="unknown", count=10, confidence=0.5)
        self.filter.classify(p)
        assert p.pattern_type == "page_number"
        assert p.confidence > 0.5

    def test_classify_roman_numeral(self):
        """罗马数字应分类为页码"""
        p = DetectedPattern(text="xvii", pattern_type="unknown", count=10, confidence=0.5)
        self.filter.classify(p)
        assert p.pattern_type == "page_number"

    def test_classify_footer_with_copyright(self):
        """版权符号应分类为页脚"""
        p = DetectedPattern(
            text="Copyright © 2023 IEEE. All rights reserved.",
            pattern_type="unknown", count=10, confidence=0.5,
        )
        self.filter.classify(p)
        assert p.pattern_type == "footer"

    def test_classify_watermark(self):
        """下载水印应分类为水印"""
        p = DetectedPattern(
            text="Downloaded on March 15, 2024. Restrictions apply.",
            pattern_type="unknown", count=10, confidence=0.5,
        )
        self.filter.classify(p)
        assert p.pattern_type == "watermark"

    def test_filter_long_content(self):
        """超长行应被判定为正文"""
        p = DetectedPattern(text="x" * 400, pattern_type="unknown", count=5, confidence=0.5)
        assert self.filter.filter_content_lines(p) is True


# =============================================================================
# Layer 3.5: CodeBlockCleaner
# =============================================================================

class TestCodeBlockCleaner:

    def test_remove_page_number_in_code(self):
        """代码块内的独立页码应被移除"""
        # CodeBlockCleaner 要求 len >= 5，所以用 "Page 42" 格式
        content = "```\nsome code\nPage 42\nmore code\n```"
        cleaner = CodeBlockCleaner()
        cleaned, count = cleaner.clean_code_blocks(content)
        assert "Page 42" not in cleaned
        assert count == 1

    def test_preserve_code_with_numbers(self):
        """代码中嵌入的数字不应被移除"""
        content = "```\ndata[42] = value;\n```"
        cleaner = CodeBlockCleaner()
        cleaned, count = cleaner.clean_code_blocks(content)
        assert "data[42]" in cleaned
        assert count == 0


# =============================================================================
# Layer 4: ContentMerger
# =============================================================================

class TestContentMerger:

    def setup_method(self):
        self.merger = ContentMerger()

    def test_merge_adjacent_code_blocks(self):
        """相邻代码块应被合并"""
        content = "```\ncode1\n```\n\n```\ncode2\n```"
        merged, code_count, table_count = self.merger.process(content)
        # 合并后应只有一对 ```
        fence_count = merged.count("```")
        assert fence_count == 2  # 一对开闭
        assert code_count == 1

    def test_merge_table_rows(self):
        """被空行分割的表格行应合并"""
        content = "| A | B |\n|---|---|\n| 1 | 2 |\n\n| 3 | 4 |"
        merged, code_count, table_count = self.merger.process(content)
        assert table_count >= 1

    def test_compress_empty_lines(self):
        """连续空行应被压缩"""
        content = "text\n\n\n\n\nmore text"
        merged, _, _ = self.merger.process(content)
        # 最多 2 个连续空行
        assert "\n\n\n\n" not in merged


# =============================================================================
# 主处理类: MarkdownCleaner (--no-llm 路径)
# =============================================================================

class TestMarkdownCleaner:

    def test_full_pipeline_no_llm(self, messy_headers_md):
        """完整清理流水线（无 LLM）应移除高频重复的页眉页脚"""
        cleaner = MarkdownCleaner(llm_client=None)
        cleaned, stats = cleaner.process(messy_headers_md)

        # 应移除版权行（高频重复，统计+启发式即可检测）
        assert "Copyright © 2023 IEEE" not in cleaned
        # 应保留正文
        assert "This chapter provides an overview" in cleaned
        # 应有统计数据
        assert stats.patterns_removed > 0
        assert stats.final_lines < stats.original_lines

    def test_preserves_code_fences(self, messy_headers_md):
        """清理不应破坏代码块边界"""
        cleaner = MarkdownCleaner(llm_client=None)
        cleaned, _ = cleaner.process(messy_headers_md)
        # 没有代码块的测试文件中也不应引入 ``` 错误
        fence_count = cleaned.count("```")
        assert fence_count % 2 == 0  # fence 总是成对的
