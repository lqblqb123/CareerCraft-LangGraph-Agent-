"""LangGraph 节点函数 —— 求职辅导工作流的每个步骤。

工作流:
    START → requirement_node ⇄ human_node（追问循环，最多 3 轮）
          → architect_node → review_node → prompt_node → export → END
    审查不再循环修正，风险提示直接附在报告中。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.types import interrupt
from loguru import logger

from app.agents.architect import ResumeOptimizerAgent
from app.agents.prompt import CareerStrategyAgent
from app.agents.requirement import CareerAdvisorAgent
from app.agents.reviewer import ResumeReviewerAgent
from app.graph.state import AgentState


def _format_resume(state: AgentState, default: str = "") -> str:
    """将优化后的简历 dict 格式化为 Markdown 文本。"""
    arch = state.get("architecture", {})
    if not arch:
        return default
    sections = []
    labels = {
        "personal_summary": "个人摘要",
        "skills_matrix": "技能矩阵",
        "work_experience": "工作/项目经历",
        "education": "教育背景",
        "additional_highlights": "附加亮点",
    }
    for key, value in arch.items():
        if value:
            label = labels.get(key, key)
            sections.append(f"## {label}\n{value}")
    return "\n\n".join(sections)


# =============================================================================
# 节点函数
# =============================================================================


def requirement_node(state: AgentState, *, llm: Any = None) -> dict[str, Any]:
    """求职顾问节点 —— 分析简历和 JD，识别差距，追问细节。

    Args:
        state: 当前工作流状态，raw_requirement=简历，requirement=JD
        llm: 语言模型实例

    Returns:
        部分状态更新，包含候选人画像、匹配度、追问问题等
    """
    agent = CareerAdvisorAgent(llm)

    rounds = state.get("question_rounds", 0)
    raw = state.get("raw_requirement", "")
    requirement = state.get("requirement", raw)
    # 提取所有历史已问问题
    all_previous = state.get("_previous_questions", "")
    current_round_q = ""
    if "## 上一轮已问过的问题" in requirement:
        parts = requirement.split("## 上一轮已问过的问题")
        q_text = parts[1].split("## 候选人回答")[0] if "## 候选人回答" in parts[1] else parts[1]
        current_round_q = q_text.strip()
        requirement = parts[0].strip()
        # 累加：历史问题 + 本轮问题
        if current_round_q:
            all_previous = (all_previous + "\n" + current_round_q).strip() if all_previous else current_round_q

    result = agent.invoke(
        raw_requirement=raw,
        requirement=requirement,
        rounds_count=rounds,
        previous_questions=all_previous,
    )

    # Build ranking summary if multiple JDs
    ranking_summary = ""
    if result.match_ranking:
        ranking_lines = ["\n\n## 多岗位匹配排名"]
        for r in result.match_ranking:
            star = "⭐" if r.jd_index == result.best_jd_index else "  "
            ranking_lines.append(
                f"{star} 岗位{r.jd_index} — {r.jd_summary} — 匹配度: {r.match_score:.0%}"
            )
        ranking_summary = "\n".join(ranking_lines)

    logger.info(
        f"[advisor_node] best_jd=#{result.best_jd_index}, "
        f"match_score={result.match_score:.2f}, "
        f"jds_ranked={len(result.match_ranking)}, "
        f"is_complete={result.is_complete}, questions={len(result.questions)}"
    )

    return {
        "requirement": result.candidate_profile + ranking_summary,
        "missing_info": result.gaps,
        "completeness_score": result.match_score,
        "questions": result.questions,
        "is_complete": result.is_complete,
        "status": "ADVISOR_ANALYZED",
    }


def human_node(state: AgentState) -> dict[str, Any]:
    """人工交互节点 —— 暂停工作流，等待候选人补充经历细节。

    使用 LangGraph 的 interrupt() 挂起执行，
    CLI 层捕获中断并展示问题，用户回答后恢复。

    Args:
        state: 当前工作流状态

    Returns:
        部分状态更新，包含合并了候选人补充信息的内容
    """
    questions = state.get("questions", [])
    logger.info(f"[human_node] Interrupting with {len(questions)} questions")

    user_response = interrupt({
        "type": "wait_human",
        "questions": questions,
        "status": "WAIT_HUMAN",
    })

    if isinstance(user_response, dict):
        answers = user_response.get("answers", "")
    elif isinstance(user_response, str):
        answers = user_response
    else:
        answers = str(user_response)

    current_requirement = state.get("requirement", "")
    asked = "\n".join(f"- {q}" for q in questions) if questions else "（无）"
    updated_requirement = (
        f"{current_requirement}\n\n"
        f"## 上一轮已问过的问题（严禁重复提问）\n{asked}\n\n"
        f"## 候选人回答\n{answers}"
    )

    logger.info("[human_node] Received candidate response, resuming workflow")

    return {
        "requirement": updated_requirement,
        "questions": [],
        "question_rounds": state.get("question_rounds", 0) + 1,
        "status": "HUMAN_INPUT_RECEIVED",
    }


def architect_node(state: AgentState, *, llm: Any = None) -> dict[str, Any]:
    """简历优化师节点 —— 基于完整画像生成优化后的简历。"""
    agent = ResumeOptimizerAgent(llm)
    requirement = state.get("requirement", "")
    raw_requirement = state.get("raw_requirement", "")
    missing_info = state.get("missing_info", [])

    extra_info = raw_requirement
    if missing_info:
        gaps_text = json.dumps(missing_info, ensure_ascii=False, indent=2)
        extra_info = f"目标岗位 JD:\n{raw_requirement}\n\n识别到的差距:\n{gaps_text}"

    result = agent.invoke(
        requirement=requirement,
        completeness_score=state.get("completeness_score", 0.0),
        extra_info=extra_info,
    )

    architecture = {
        "personal_summary": result.personal_summary,
        "skills_matrix": result.skills_matrix,
        "work_experience": result.work_experience,
        "education": result.education,
        "additional_highlights": result.additional_highlights,
    }

    logger.info("[optimizer_node] Resume optimized")

    return {
        "architecture": architecture,
        "question_rounds": 0,
        "status": "RESUME_OPTIMIZED",
    }


def review_node(state: AgentState, *, llm: Any = None) -> dict[str, Any]:
    """简历审查节点 —— 审查优化后的简历质量。

    检查真实性、ATS 兼容性、措辞质量、篇幅控制、STAR 完整性。

    Args:
        state: 当前工作流状态
        llm: 语言模型实例

    Returns:
        部分状态更新，包含审查结果
    """
    agent = ResumeReviewerAgent(llm)
    requirement = state.get("requirement", "")
    architecture = _format_resume(state, default="（暂无优化简历）")

    result = agent.invoke(
        requirement=requirement,
        architecture=architecture,
    )

    feedback_text = "\n".join(
        [f"- 问题: {i}" for i in result.issues]
        + [f"- 建议: {s}" for s in result.suggestions]
    )

    logger.info(
        f"[review_node] passed={result.passed}, score={result.score:.2f}"
    )

    return {
        "review_feedback": feedback_text,
        "status": "REVIEWED",
    }


def prompt_node(state: AgentState, *, llm: Any = None) -> dict[str, Any]:
    """求职策略节点 —— 生成 Cover Letter、社交文案、提升建议。

    Args:
        state: 当前工作流状态
        llm: 语言模型实例

    Returns:
        部分状态更新，包含求职配套文案
    """
    agent = CareerStrategyAgent(llm)
    requirement = state.get("requirement", "")
    architecture = _format_resume(state)

    result = agent.invoke(
        requirement=requirement,
        architecture=architecture,
        jd_text=state.get("raw_requirement", ""),
    )

    logger.info("[strategy_node] Generated growth plan")

    return {
        "growth_plan": result.growth_plan,
        "status": "STRATEGY_GENERATED",
    }


def export_node(state: AgentState) -> dict[str, Any]:
    """导出节点 —— 标记工作流完成。

    实际的文件导出由 CLI/导出层处理。

    Args:
        state: 当前工作流状态

    Returns:
        部分状态更新，标记完成
    """
    logger.info("[export_node] Workflow complete, ready for export")
    return {
        "status": "COMPLETED",
    }
