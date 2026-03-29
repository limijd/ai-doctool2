"""
aidoc_llm 单元测试
===================

测试配置加载、客户端工厂、JSON 提取等核心功能。
LLM 连接测试在 test_integration.py 中。
"""

import pytest
from aidoc_llm import (
    load_config,
    extract_json,
    create_llm_client_from_config,
    OllamaClient,
    OpenAIClient,
    add_llm_args,
    create_llm_client,
)


# =============================================================================
# extract_json
# =============================================================================

class TestExtractJson:
    """JSON 提取测试"""

    def test_pure_json(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_text(self):
        text = 'Here is the result: {"answer": "A", "confidence": 0.9} end.'
        result = extract_json(text)
        assert result["answer"] == "A"
        assert result["confidence"] == 0.9

    def test_no_json(self):
        assert extract_json("no json here") is None

    def test_empty_string(self):
        assert extract_json("") is None

    def test_none_input(self):
        assert extract_json(None) is None

    def test_malformed_json(self):
        assert extract_json("{broken json") is None

    def test_json_with_chinese(self):
        result = extract_json('{"summary": "这是摘要", "keywords": ["关键字"]}')
        assert result["summary"] == "这是摘要"


# =============================================================================
# 工厂函数
# =============================================================================

class TestFactory:
    """客户端工厂函数测试"""

    def test_default_creates_ollama(self):
        """默认创建 Ollama 客户端"""
        client = create_llm_client_from_config(api="ollama")
        assert isinstance(client, OllamaClient)

    def test_openai_without_key_not_available(self):
        """OpenAI 无 key 时 available=False"""
        import os
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            client = create_llm_client_from_config(api="openai", api_key="")
            assert isinstance(client, OpenAIClient)
            assert not client.available
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_cli_args_integration(self):
        """add_llm_args + create_llm_client 联动"""
        import argparse
        parser = argparse.ArgumentParser()
        add_llm_args(parser)
        args = parser.parse_args(["--no-llm"])
        client = create_llm_client(args)
        assert client is None

    def test_cli_args_with_model(self):
        """CLI 指定模型"""
        import argparse
        parser = argparse.ArgumentParser()
        add_llm_args(parser)
        args = parser.parse_args(["--model", "llama3.2:3b"])
        client = create_llm_client(args)
        assert client.model == "llama3.2:3b"


# =============================================================================
# 配置文件加载
# =============================================================================

class TestConfig:
    """配置加载测试"""

    def test_load_config_returns_dict(self):
        """load_config 总是返回字典"""
        config = load_config()
        assert isinstance(config, dict)
