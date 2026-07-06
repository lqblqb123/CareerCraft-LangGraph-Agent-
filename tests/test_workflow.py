"""Tests for the LangGraph workflow and supporting modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.graph.state import AgentState, create_initial_state
from app.graph.router import (
    route_after_human,
    route_after_requirement,
    route_after_review,
)


class TestAgentState:
    """Tests for AgentState and state creation."""

    def test_create_initial_state(self):
        """Test creating an initial workflow state."""
        state = create_initial_state("我想做一个外卖平台")
        assert state["raw_requirement"] == "我想做一个外卖平台"
        assert state["requirement"] == "我想做一个外卖平台"
        assert state["questions"] == []
        assert state["missing_info"] == []
        assert state["completeness_score"] == 0.0
        assert state["is_complete"] is False
        assert state["architecture"] == {}
        assert state["question_rounds"] == 0
        assert state["status"] == "START"

    def test_state_keys_exist(self):
        """Test that all expected keys are present in the initial state."""
        state = create_initial_state("test")
        expected_keys = [
            "raw_requirement",
            "requirement",
            "questions",
            "missing_info",
            "completeness_score",
            "is_complete",
            "architecture",
            "review_feedback",
            "growth_plan",
            "question_rounds",
            "status",
        ]
        for key in expected_keys:
            assert key in state, f"Missing key: {key}"


class TestRouter:
    """Tests for conditional edge routing."""

    def test_route_after_requirement_needs_clarification(self):
        """Test routing to human_node when requirement is incomplete."""
        state = create_initial_state("test")
        state["is_complete"] = False
        state["questions"] = ["问题1", "问题2"]
        state["completeness_score"] = 0.4

        result = route_after_requirement(state)
        assert result == "human_node"

    def test_route_after_requirement_ready(self):
        """Test routing to architect_node when requirement is complete."""
        state = create_initial_state("test")
        state["is_complete"] = True
        state["questions"] = []
        state["completeness_score"] = 0.85

        result = route_after_requirement(state)
        assert result == "architect_node"

    def test_route_after_human(self):
        """Test routing after human input is always back to requirement."""
        state = create_initial_state("test")
        result = route_after_human(state)
        assert result == "requirement_node"

    def test_route_after_review_always_prompt(self):
        """Review always proceeds to prompt_node (no revision loop)."""
        state = create_initial_state("test")
        assert route_after_review(state) == "prompt_node"


class TestWorkflow:
    """Tests for workflow compilation."""

    def test_create_workflow(self, mock_llm):
        """Test that a workflow can be compiled."""
        from app.graph.workflow import create_workflow

        workflow = create_workflow(llm=mock_llm)
        assert workflow is not None
        # The compiled graph should have a 'get_graph' method or similar
        assert hasattr(workflow, "stream") or hasattr(workflow, "invoke")


class TestExporter:
    """Tests for the Markdown exporter."""

    def test_exporter_creation(self):
        """Test that the exporter can be created."""
        from app.exporter.markdown import MarkdownExporter

        exporter = MarkdownExporter()
        assert exporter is not None
        assert exporter.env is not None

    def test_prepare_context(self, sample_state):
        """Test context preparation for templates."""
        from app.exporter.markdown import MarkdownExporter

        exporter = MarkdownExporter()
        context = exporter._prepare_context(sample_state)
        assert "personal_summary" in context
        assert "review_feedback" in context
        assert "growth_plan" in context
        assert context["personal_summary"]
        assert "generation_time" in context

    def test_export_all(self, sample_state, tmp_path):
        """Test exporting all documents."""
        from app.exporter.markdown import MarkdownExporter

        exporter = MarkdownExporter()
        output_dir = str(tmp_path / "output")
        files = exporter.export_all(sample_state, output_dir)

        assert len(files) > 0
        for f in files:
            assert tmp_path / "output" / Path(f).name
