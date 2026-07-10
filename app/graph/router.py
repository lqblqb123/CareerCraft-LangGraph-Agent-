"""
LangGraph 工作流的条件边路由逻辑。

每个路由函数检查当前状态，并返回下一个要执行的节点名称。
这种设计实现了工作流的动态分支控制。
"""

from __future__ import annotations

from loguru import logger

from app.graph.state import AgentState


def route_after_requirement(state: AgentState) -> str:
    """需求分析后路由：Agent 判断完整则继续，不完整则追问，最多 3 轮强制截断。"""
    is_complete = state.get("is_complete", False)
    questions = state.get("questions", [])
    completeness = state.get("completeness_score", 0.0)
    question_rounds = state.get("question_rounds", 0)

    # 硬上限：追问达到 3 轮，不管 Agent 怎么判断都强制进入优化
    if question_rounds >= 3:
        logger.info(f"[router] Max rounds ({question_rounds}), forcing architect")
        return "architect_node"

    # Agent 认为不完整 + 有问题 + 评分不够 → 继续追问
    if not is_complete and questions and completeness < 0.7:
        logger.info(
            f"[router] Need clarification: round={question_rounds + 1}, "
            f"completeness={completeness:.2f}, questions={len(questions)}"
        )
        return "human_node"

    # Agent 判断完整 / 没问题 / 评分够了 → 进入优化
    logger.info("[router] Agent says complete, proceeding to architect")
    return "architect_node"

def route_after_human(state: AgentState) -> str:
    """人回答后路由：满 3 轮直接进优化，否则回分析。"""
    question_rounds = state.get("question_rounds", 0)
    if question_rounds >= 3:
        logger.info(f"[router] Max rounds ({question_rounds}), forcing architect")
        return "architect_node"
    logger.info("[router] Returning to requirement analysis")
    return "requirement_node"


