"""Shared test fixtures and utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage


@pytest.fixture
def sample_requirement() -> str:
    """A sample resume text for testing."""
    return "张三，3年Python后端开发经验，熟悉FastAPI和Django，在XX公司负责核心API开发。"


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """A sample career coach workflow state for testing."""
    return {
        "raw_requirement": "张三的简历\nPython后端 3年经验...",
        "requirement": "候选人画像：3年Python后端开发，熟悉FastAPI...",
        "questions": [],
        "missing_info": [
            {"category": "quantify", "gap": "缺少量化数据", "importance": "medium"}
        ],
        "completeness_score": 0.85,
        "is_complete": True,
        "architecture": {
            "personal_summary": "3年Python后端开发经验",
            "skills_matrix": "精通: Python, FastAPI\n熟练: Docker, PostgreSQL",
            "work_experience": "某公司 后端开发 2022-2025\n- 主导开发了XX系统",
            "education": "本科 计算机科学",
            "additional_highlights": "GitHub 500 stars",
        },
        "review_feedback": "",
        "growth_plan": "",
        "status": "RESUME_OPTIMIZED",
    }


@pytest.fixture
def mock_llm() -> BaseChatModel:
    """Create a mock LLM that returns a predefined AIMessage.

    For unit tests that don't need real API calls.
    """
    mock = MagicMock(spec=BaseChatModel)
    mock.invoke.return_value = AIMessage(content='{"passed": true, "issues": [], "suggestions": [], "score": 0.9}')
    return mock
