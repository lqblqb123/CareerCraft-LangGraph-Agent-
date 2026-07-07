"""简历审查员 Agent — 检查优化后简历的真实性、ATS 兼容性、措辞质量。

审查结果作为风险提示附在报告中，不自动修正。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class ResumeReviewResult(BaseModel):
    """简历审查员的结构化输出 —— 审查不循环，结果作为风险提示附在报告中。"""

    # ↓ 拼入 review_feedback → state["review_feedback"] → 模板渲染风险提示 + full_report 第二章
    issues: list[str] = Field(
        default_factory=list,
        description="发现的问题列表（每项注明具体位置和问题）",
    )

    # ↓ 拼入 review_feedback → 同上
    suggestions: list[str] = Field(
        default_factory=list,
        description="修改建议列表（每条具体可执行）",
    )


class ResumeReviewerAgent:
    """简历审查员 Agent —— 严格审查，防止夸大，确保可投递。

    审查维度：
    1. 真实性：是否有夸大或可能被面试官质疑的描述
    2. ATS 兼容：关键词覆盖和格式能否通过机器人筛选
    3. 措辞质量：动词是否有力、描述是否简洁
    4. 篇幅控制：是否过长或过短
    5. STAR 完整性：每条经历是否具备完整的 STAR 四要素
    """

    name = "ResumeReviewerAgent"

    SYSTEM_PROMPT = """你是一个严格的简历审查专家和面试官视角的评估者。你的工作是站在招聘方的角度，审查优化后的简历是否足够有竞争力。

## 审查维度

### 1. 真实性审查
- 每条数据是否看起来可信？如果面试官深挖这段经历，候选人能否对答如流？
- 是否存在"参与"变"主导"的过度包装？（这会引起面试官警觉）
- 量化数据是否有明显的夸大痕迹？
- 技术栈描述是否与项目规模匹配？

### 2. ATS 兼容性
- 目标岗位的关键词是否在简历中出现（尤其是技能部分）？
- 是否有机器人筛选可能漏掉的关键信息？
- 格式是否简洁清晰（避免表格、图片、特殊字符）？

### 3. 措辞质量
- 是否避免了"负责"、"参与"等弱动词？
- 每句开头是否以强有力的动作动词开始？
- 是否有冗余或空洞的表述？
- 专业术语使用是否准确？

### 4. 篇幅与结构
- 简历是否控制在 1-2 页 A4 以内？
- 最重要的信息是否在前三分之一？
- 每段经历的要点是否控制在 2-4 条？

### 5. STAR 完整性
- 每条经历是否包含：情境（可选）、任务（可选）、行动（必须）、结果（必须）
- 结果是否量化？

## 输出格式
- issues: 发现的问题列表（每项注明具体位置和问题）
- suggestions: 具体的修改建议（每条可执行、可直接用于修改）"""

    HUMAN_TEMPLATE = """请审查以下优化后的简历。

## 候选人原始信息（作为真实性基准）
{requirement}

## 优化后的简历
{architecture}

请逐项审查，输出 issues（问题）和 suggestions（建议）。要严格但建设性。"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.debug("ResumeReviewerAgent initialized")

    def invoke(
        self,
        requirement: str = "",
        architecture: str = "",
    ) -> ResumeReviewResult:
        human_prompt = self.HUMAN_TEMPLATE.format(
            requirement=requirement,
            architecture=architecture,
        )

        logger.info("ResumeReviewerAgent reviewing")
        result = self._call_llm(human_prompt)
        logger.info(
            f"ResumeReviewerAgent: issues={len(result.issues)}, "
            f"suggestions={len(result.suggestions)}"
        )
        return result

    def _call_llm(self, human_prompt: str) -> ResumeReviewResult:
        from datetime import datetime
        now = datetime.now().strftime("%Y 年 %m 月 %d 日")
        messages = [
            SystemMessage(content=f"当前日期: {now}\n\n{self.SYSTEM_PROMPT}"),
            HumanMessage(content=human_prompt),
        ]
        structured_llm = self.llm.with_structured_output(ResumeReviewResult)
        return structured_llm.invoke(messages)
