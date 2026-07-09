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
    # 取完整 requirement（含历史回答），Agent 需要看到最新上下文
    requirement = state.get("requirement", raw)

    result = agent.invoke(
        raw_requirement=raw,
        requirement=requirement,
        rounds_count=rounds,
        previous_questions=state.get("previous_questions", ""),
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
    asked = "\n".join(f"- {q}" for q in questions) if questions else ""
    # 问题+答案一起追加，避免"是/对"等简短回答脱离上下文
    qa_text = f"问: {'; '.join(questions)}\n答: {answers}" if questions else f"答: {answers}"
    updated_requirement = (
        f"{current_requirement}\n\n## 本轮问答\n{qa_text}"
    )

    # 将本轮问题累加到历史问题池
    prev = state.get("previous_questions", "")
    if asked:
        tag = f"第{state.get('question_rounds', 0) + 1}轮:\n{asked}"
        new_prev = (prev + "\n\n" + tag).strip() if prev else tag
    else:
        new_prev = prev

    logger.info("[human_node] Received candidate response, resuming workflow")

    return {
        "requirement": updated_requirement,
        "previous_questions": new_prev,
        "questions": [],
        "question_rounds": state.get("question_rounds", 0) + 1,
        "status": "HUMAN_INPUT_RECEIVED",
    }


def _extract_education(raw_text: str) -> str:
    """从原始简历中直接提取教育背景，不经过 LLM。

    用常见关键词（学校/学院/大学/学历/专业/毕业）匹配相关行，
    避免 LLM 篡改日期、校名等硬数据。
    """
    import re

    resume_part = raw_text.split("## 目标岗位描述")[0] if "## 目标岗位描述" in raw_text else raw_text
    resume_part = resume_part.replace("## 候选人简历\n", "").strip()

    lines = resume_part.split("\n")
    edu_lines = []
    capturing = False
    edu_keywords = ["教育", "学历", "学校", "大学", "学院", "毕业", "专业", "本科", "硕士", "博士", "专科", "学位"]

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if capturing:
                break  # 空行结束教育模块
            continue
        # 检测教育模块开头
        if any(kw in line_stripped for kw in edu_keywords):
            capturing = True
        if capturing:
            edu_lines.append(line_stripped)

    if edu_lines:
        return "\n".join(edu_lines)

    # Fallback: 查找包含教育关键词的行
    for line in lines:
        if any(kw in line.strip() for kw in edu_keywords):
            edu_lines.append(line.strip())
    return "\n".join(edu_lines) if edu_lines else "（未从简历中识别到教育背景）"


def _validate_hard_data(raw_text: str, optimized: dict[str, str]) -> list[str]:
    """验证优化后简历中的硬数据是否与原始一致。

    检测项：日期格式、公司名、学校名、邮箱。返回被篡改的告警列表。
    """
    import re

    warnings = []
    # 从简历原文中提取硬数据（取 ## 候选人简历 之后的部分）
    resume_part = raw_text.split("## 目标岗位描述")[0] if "## 目标岗位描述" in raw_text else raw_text
    resume_part = resume_part.replace("## 候选人简历\n", "").strip()

    # 1. 日期校验：提取原文中的日期模式
    patterns = [r"\d{4}[.年]\d{1,2}[月]?", r"\d{4}[.年-]\d{1,2}[月]?\s*[-~—至到]\s*\d{4}[.年-]?\d{0,2}[月]?", r"\d{4}\.\d{1,2}"]
    original_dates = set()
    for pat in patterns:
        for m in re.finditer(pat, resume_part):
            original_dates.add(m.group())

    # 2. 公司和学校名提取：常见后缀
    company_patterns = [r"[一-鿿]{2,20}(?:公司|有限|集团|科技|技术|软件|信息|股份|电子|电气|自动化|新能源|设备|光电|通信|网络)", r"[一-鿿]{2,20}(?:大学|学院|学校)"]
    original_entities = set()
    for pat in company_patterns:
        for m in re.finditer(pat, resume_part):
            original_entities.add(m.group())

    # 3. 邮箱校验
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", resume_part)
    original_entities.update(emails)

    # 4. 电话校验
    phones = re.findall(r"1[3-9]\d{9}", resume_part)
    original_entities.update(phones)

    # 拼接所有优化后文本
    optimized_text = " ".join(optimized.values())

    # 检查日期
    for d in original_dates:
        if d not in optimized_text:
            warnings.append(f"日期 '{d}' 在优化后丢失或被改写")

    # 检查实体
    for e in original_entities:
        if e not in optimized_text:
            warnings.append(f"'{e}' 在优化后丢失或被改写")

    return warnings


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
        "education": _extract_education(raw_requirement),
        "additional_highlights": result.additional_highlights,
    }

    # 硬数据防篡改：检查优化后是否保留了原文关键数据
    hard_data_warnings = _validate_hard_data(raw_requirement, architecture)
    if hard_data_warnings:
        for w in hard_data_warnings:
            logger.warning(f"[optimizer_node] 硬数据校验失败: {w}")

    logger.info("[optimizer_node] Resume optimized"
                + (f" (硬数据告警: {len(hard_data_warnings)} 处)" if hard_data_warnings else ""))

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

    logger.info(f"[review_node] feedback={len(result.feedback)} chars")

    return {
        "review_feedback": result.feedback,
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
