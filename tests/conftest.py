"""
aidoc 测试公共 fixtures
========================

提供 mock LLM 客户端、样本文件路径、临时输出目录等。

标记约定:
  - 无标记: 离线单元测试，不依赖外部服务
  - @pytest.mark.integration: 需要 Ollama + qwen3:8b
"""

import sys
import pytest
from pathlib import Path

# 将项目根目录加入 sys.path，使 tests/ 下能 import aidoc_* 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_DIR = Path(__file__).parent / "samples"


# =============================================================================
# pytest 标记注册
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要 Ollama 运行 (qwen3:8b)")


# =============================================================================
# 样本文件 fixtures
# =============================================================================

@pytest.fixture
def samples_dir():
    """测试样本目录"""
    return SAMPLES_DIR


@pytest.fixture
def messy_headers_md():
    """包含重复页眉页脚的 Markdown 内容"""
    return (SAMPLES_DIR / "messy_headers.md").read_text(encoding="utf-8")


@pytest.fixture
def flat_hierarchy_md():
    """标题层级全部退化为 ## 的 Markdown 内容"""
    return (SAMPLES_DIR / "flat_hierarchy.md").read_text(encoding="utf-8")


@pytest.fixture
def broken_codeblocks_md():
    """代码块边界错位的 Markdown 内容"""
    return (SAMPLES_DIR / "broken_codeblocks.md").read_text(encoding="utf-8")


@pytest.fixture
def structured_doc_md():
    """干净的结构化文档 Markdown 内容"""
    return (SAMPLES_DIR / "structured_doc.md").read_text(encoding="utf-8")


# =============================================================================
# Mock LLM 客户端
# =============================================================================

class MockLLMClient:
    """
    Mock LLM 客户端，用于离线测试。

    返回预设的响应，模拟 LLM 分类/推断行为。
    可通过 response_map 自定义不同 prompt 关键词对应的响应。
    """

    def __init__(self, default_response: str = "", response_map: dict = None):
        self.model = "mock-model"
        self.available = True
        self.call_count = 0
        self.last_prompt = ""
        self.default_response = default_response
        self.response_map = response_map or {}

    def generate(self, prompt: str, system: str = "", temperature=None, max_tokens=2048) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        # 按关键词匹配预设响应
        for keyword, response in self.response_map.items():
            if keyword in prompt:
                return response
        return self.default_response

    def check_connection(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "Mock"


@pytest.fixture
def mock_llm():
    """基础 mock LLM，返回空响应"""
    return MockLLMClient()


@pytest.fixture
def mock_llm_classifier():
    """模拟分类行为的 mock LLM，对页眉页脚返回分类 JSON"""
    return MockLLMClient(
        default_response='{"1": "CONTENT"}',
        response_map={
            "Copyright": '{"1": "FOOTER"}',
            "Standard for": '{"1": "HEADER"}',
            "Downloaded": '{"1": "WATERMARK"}',
        },
    )


@pytest.fixture
def mock_llm_hierarchy():
    """模拟层级推断的 mock LLM，返回数字层级"""
    return MockLLMClient(
        default_response="2",
        response_map={
            "IEEE Standard": "1",
            "Introduction": "2",
            "Abstract": "2",
            "Overview": "2",
        },
    )


# =============================================================================
# Ollama 集成测试 fixture
# =============================================================================

@pytest.fixture
def ollama_client():
    """
    真实 Ollama 客户端 (qwen3:8b)。

    仅在 integration 测试中使用，自动跳过不可用的环境。
    """
    from aidoc_llm import OllamaClient
    client = OllamaClient(model="qwen3:8b")
    if not client.available:
        pytest.skip("Ollama qwen3:8b 不可用")
    return client


# =============================================================================
# 临时输出目录
# =============================================================================

@pytest.fixture
def output_dir(tmp_path):
    """临时输出目录，测试结束自动清理"""
    return tmp_path
