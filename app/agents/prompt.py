"""求职策略 Agent — 生成能力差距分析与提升建议。"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class CareerStrategyResult(BaseModel):
    """求职策略 Agent 的结构化输出。"""

    growth_plan: str = Field(
        default="",
        description="能力差距分析与提升建议：技能缺失、推荐资源、预估时间",
    )


class CareerStrategyAgent:
    """求职策略 Agent —— 生成能力差距分析与提升建议。"""

    name = "CareerStrategyAgent"

    SYSTEM_PROMPT = """你是一个资深的职业规划顾问。基于候选人与目标岗位的差距，生成能力提升计划。

## 输出内容
- 需要补充的技能（按优先级 P0/P1/P2 排列）
- 每项技能的推荐学习资源（书、课程、开源项目）
- 预估学习时间和达到面试标准的路径
- 可以"包装"哪些现有经历来弥补部分差距

格式要求：Markdown 格式，分 P0/P1/P2 三个优先级。"""

    HUMAN_TEMPLATE = """请基于以下信息生成能力差距分析与提升建议。

## 候选人信息
{requirement}

## 优化后的简历
{architecture}

## 目标岗位
{jd_text}"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.debug("CareerStrategyAgent initialized")

    def invoke(
        self,
        requirement: str = "",
        architecture: str = "",
        jd_text: str = "",
    ) -> CareerStrategyResult:
        human_prompt = self.HUMAN_TEMPLATE.format(
            requirement=requirement,
            architecture=architecture or "（见候选人信息）",
            jd_text=jd_text or "（见候选人信息）",
        )

        logger.info("CareerStrategyAgent generating growth plan")
        return self._call_llm(human_prompt)

    def _call_llm(self, human_prompt: str) -> CareerStrategyResult:
        from datetime import datetime
        now = datetime.now().strftime("%Y 年 %m 月 %d 日")
        messages = [
            SystemMessage(content=f"当前日期: {now}\n\n{self.SYSTEM_PROMPT}"),
            HumanMessage(content=human_prompt),
        ]
        structured_llm = self.llm.with_structured_output(CareerStrategyResult)
        return structured_llm.invoke(messages)
