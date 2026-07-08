"""LangGraph workflow — CareerCraft 工作流图。"""

from __future__ import annotations

from typing import Any, Callable

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.graph.nodes import (
    architect_node,
    export_node,
    human_node,
    prompt_node,
    requirement_node,
    review_node,
)
from app.graph.router import (
    route_after_human,
    route_after_requirement,
)
from app.graph.state import AgentState


def create_workflow(
    llm: Any = None,
    checkpointer: Any = None,
) -> StateGraph:
    """Build and compile the CareerCraft LangGraph workflow."""
    if llm is None:
        from app.config.llm import create_llm
        llm = create_llm()
        logger.info("Created default LLM for workflow")

    if checkpointer is None:
        db_dir = Path("data")
        db_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_dir / "checkpoints.db"), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("Using SqliteSaver for workflow checkpoints")

    workflow = StateGraph(AgentState)

    workflow.add_node("requirement_node", _with_llm(requirement_node, llm))
    workflow.add_node("human_node", human_node)
    workflow.add_node("architect_node", _with_llm(architect_node, llm))
    workflow.add_node("review_node", _with_llm(review_node, llm))
    workflow.add_node("prompt_node", _with_llm(prompt_node, llm))
    workflow.add_node("export_node", export_node)

    # Entry
    workflow.set_entry_point("requirement_node")

    # Questioning loop
    workflow.add_conditional_edges(
        "requirement_node", route_after_requirement,
        {"human_node": "human_node", "architect_node": "architect_node"},
    )
    workflow.add_conditional_edges(
        "human_node", route_after_human,
        {"requirement_node": "requirement_node", "architect_node": "architect_node"},
    )

    # Architect → review → prompt → export
    workflow.add_edge("architect_node", "review_node")
    workflow.add_edge("review_node", "prompt_node")
    workflow.add_edge("prompt_node", "export_node")
    workflow.add_edge("export_node", END)

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("Workflow compiled successfully")
    return app


def _with_llm(node_func: Callable, llm: Any) -> Callable:
    """闭包注入 LLM 实例到节点函数。"""
    def wrapped(state: AgentState) -> dict[str, Any]:
        return node_func(state, llm=llm)
    wrapped.__name__ = node_func.__name__
    return wrapped
