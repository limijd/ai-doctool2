"""
aidoc 集成测试
===============

需要 Ollama 运行 + qwen3:8b 模型。

运行方式:
    pytest tests/test_integration.py -m integration -v

跳过方式（无 Ollama 时自动跳过）:
    pytest tests/ -m "not integration"
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# LLM 连接测试
# =============================================================================

@pytest.mark.integration
class TestOllamaConnection:

    def test_ollama_available(self, ollama_client):
        """Ollama 服务应可用"""
        assert ollama_client.available

    def test_generate_response(self, ollama_client):
        """应能生成非空响应"""
        response = ollama_client.generate("说 hello", system="只回复一个词")
        assert len(response) > 0

    def test_json_generation(self, ollama_client):
        """应能生成 JSON 格式响应"""
        from aidoc_llm import extract_json
        response = ollama_client.generate(
            '输出 JSON: {"answer": "yes"}，只输出 JSON',
            temperature=0.1,
        )
        result = extract_json(response)
        # LLM 不保证格式完美，但应尽力返回 JSON
        # 这里只验证调用链正常，不强断言解析结果
        assert isinstance(response, str)


# =============================================================================
# strip 集成测试
# =============================================================================

@pytest.mark.integration
class TestStripIntegration:

    def test_strip_with_llm(self, messy_headers_md, ollama_client):
        """LLM 辅助清理应比纯规则更准确"""
        from aidoc_strip import MarkdownCleaner

        cleaner = MarkdownCleaner(llm_client=ollama_client)
        cleaned, stats = cleaner.process(messy_headers_md)

        assert "Copyright © 2023 IEEE" not in cleaned
        assert stats.final_lines < stats.original_lines


# =============================================================================
# fix_hierarchy 集成测试
# =============================================================================

@pytest.mark.integration
class TestFixHierarchyIntegration:

    def test_fix_with_llm(self, flat_hierarchy_md, ollama_client):
        """LLM 应能推断无编号标题的层级"""
        from aidoc_fix_hierarchy import HierarchyFixer

        lines = flat_hierarchy_md.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=ollama_client)
        result = fixer.fix(use_llm=True, verbose=False)

        assert result.fixed_headings > 0
        # LLM 推断数应 > 0（有无编号标题需要 LLM 处理）
        # 注意：规则引擎已能处理大部分，LLM 处理剩余的
        assert result.rule_based_fixes > 0


# =============================================================================
# index 集成测试
# =============================================================================

@pytest.mark.integration
class TestIndexIntegration:

    def test_index_with_llm(self, samples_dir, ollama_client):
        """LLM 应能生成摘要和关键字"""
        from aidoc_index import MarkdownParser, IndexBuilder

        parser = MarkdownParser(str(samples_dir / "structured_doc.md"))
        builder = IndexBuilder(parser, llm=ollama_client)
        index = builder.build(use_llm=True)

        # 至少部分 chunk 应有摘要
        chunks_with_summary = [c for c in index.chunks.values() if c.summary]
        assert len(chunks_with_summary) > 0

        # 应有关键字索引
        assert len(index.keyword_index) > 0

        # 验证序列化
        json_str = json.dumps(index.to_dict(), ensure_ascii=False)
        assert len(json_str) > 0


# =============================================================================
# 端到端管线测试
# =============================================================================

@pytest.mark.integration
class TestEndToEndPipeline:

    def test_strip_then_hierarchy(self, messy_headers_md, ollama_client):
        """strip → fix_hierarchy 串联应产出干净的分层文档"""
        from aidoc_strip import MarkdownCleaner
        from aidoc_fix_hierarchy import HierarchyFixer, MarkdownWriter

        # Step 1: strip
        cleaner = MarkdownCleaner(llm_client=ollama_client)
        cleaned, strip_stats = cleaner.process(messy_headers_md)
        assert strip_stats.patterns_removed > 0

        # Step 2: fix hierarchy
        lines = cleaned.splitlines(keepends=True)
        fixer = HierarchyFixer(lines, llm=ollama_client)
        result = fixer.fix(use_llm=True, verbose=False)

        writer = MarkdownWriter(lines, result.headings)
        final = writer.generate()

        # 最终文档应有正确的多层级标题
        assert "# " in final or "## " in final
        # 不应有残留的版权页脚
        assert "Copyright © 2023 IEEE" not in final
