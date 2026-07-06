"""Agent state definitions for the LangGraph workflow."""

from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """LangGraph workflow state — 所有节点共享的中央状态。"""

    # 求职顾问阶段
    raw_requirement: str                # 简历原文，不变architect,prompt需要原始DJ从这获取
    # 初始：简历 + JD 原文
    # 追问时：被 human_node 追加你的回答,
    # 分析后：被 requirement_node 覆盖为 LLM 提炼的"候选人画像"
    requirement: str                    # 候选人画像 / JD，不断被补充
    questions: list[str]                # 待追问候选人的问题
    # 求职顾问发现你简历和 JD 的差距列表：[
    #     {"category": "高并发", "gap": "JD 要求高并发经验，简历未体现", "importance": "high"},
    #     {"category": "量化",   "gap": "项目成果缺少 QPS/DAU 数据",      "importance": "medium"},
    # ]
    missing_info: list[dict[str, str]]  # 差距列表，给优化师参考
    completeness_score: float           # 匹配度 0-1
    is_complete: bool                   # 信息是否完整

    # 简历优化阶段
    architecture: dict[str, str]        # 优化后简历各部分

    # 审查阶段
    review_feedback: str                # 审查反馈文本（作为风险提示导出）

    # 求职策略阶段
    growth_plan: str                    # 能力提升建议

    # 流程控制
    question_rounds: int                # 追问轮数计数，最大 3 轮
    status: Annotated[str, lambda a, b: b]  # 当前阶段标记（并发写入时取最后一个）


def create_initial_state(raw_requirement: str) -> AgentState:
    """创建初始状态。"""
    return AgentState(
        raw_requirement=raw_requirement,
        requirement=raw_requirement,
        questions=[],
        missing_info=[],
        completeness_score=0.0,
        is_complete=False,
        architecture={},
        review_feedback="",
        growth_plan="",
        question_rounds=0,
        status="START",
    )
