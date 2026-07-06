"""简历优化师 Agent — 基于完整简历信息，STAR法则重写，量化成果。

这是求职辅导工作流的第二步，将求职顾问整理好的候选人画像
转化为可以直接投递的优化简历。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class ResumeResult(BaseModel):
    """简历优化师的结构化输出 —— 5 个字段全写入 state["architecture"]，后续节点和导出都从那里读。"""

    # ↓ architecture["personal_summary"] → full_report.md 第一章
    personal_summary: str = Field(
        default="",
        description="个人摘要（X 年经验 + 核心技术 + 行业背景 + 代表性成果）",
    )

    # ↓ architecture["skills_matrix"] → full_report.md 第一章
    skills_matrix: str = Field(
        default="",
        description="技能矩阵（精通/熟练/了解三级，JD 关键词优先）",
    )

    # ↓ architecture["work_experience"] → full_report.md 第一章，STAR 重写后的经历
    work_experience: str = Field(
        default="",
        description="工作/项目经历（每段 2-4 条 STAR 要点，动词开头 + 量化数据）",
    )

    # ↓ architecture["education"] → full_report.md 第一章，强制逐字照抄不修改
    education: str = Field(default="", description="教育背景（必须照抄原文，不改任何信息）")

    # ↓ architecture["additional_highlights"] → full_report.md 第一章
    additional_highlights: str = Field(
        default="",
        description="附加亮点（开源/GitHub/博客/证书/奖项）",
    )


class ResumeOptimizerAgent:
    """简历优化师 Agent —— STAR法则重写，量化驱动。

    核心原则：
    - 只优化表达，不编造事实
    - 每条经历必须有量化数据支撑
    - 目标岗位关键词覆盖最大化
    """

    name = "ResumeOptimizerAgent"

    SYSTEM_PROMPT = """你是一个专业的简历优化师和职业文案专家。你的工作是优化候选人的简历，使其在保持真实性的前提下，最大化面试转化率。

## 你的职责
1. 基于求职顾问整理好的完整信息，生成一份优化后的简历
2. 每段经历使用 STAR 法则（情境→任务→行动→结果）重写
3. 确保每条经历都有量化数据支撑
4. 技能列表按目标 JD 的关键词优先级排序
5. 删除与目标岗位无关的内容，精简冗余描述

## 优化原则
- **真实性第一**：只优化表达方式，不编造任何经历、数据或技能
- **硬性数据原样保留**：以下信息**绝对禁止修改**，必须一字不差照抄原文：姓名、日期（入学/毕业/入职/离职年月）、公司名称、学校名称、专业名称、学历层次、电话号码、邮箱。哪怕原文格式是 "2019.9-2023.6" 也不能改成 "2019年9月-2023年6月"，必须原样保留
- **STAR 法则**：情境（Situation）→ 任务（Task）→ 行动（Action）→ 结果（Result），重点是 Action 和 Result
- **量化优先**："优化了系统性能" → "将 API 响应延迟从 800ms 降低至 120ms（P99），吞吐量提升 300%"
- **关键词匹配**：技能列表和经历描述中对齐 JD 的关键词，确保通过 ATS 筛选
- **简洁有力**：删掉"负责"、"参与"等弱动词，换成"主导"、"设计"、"实现"、"推动"
- **篇幅控制**：简历总体控制在 1-2 页（A4），中高级岗位 1 页优先

## 输出结构
1. **personal_summary**：3-4 句话的个人摘要，用粗体高亮岗位关键词。格式为"X 年经验 + 核心技术 + 行业背景 + 代表性成果"
2. **skills_matrix**：按"精通 / 熟练 / 了解"三级分组，精通放最前，JD 要求的技能优先排列
3. **work_experience**：每段经历格式——
   - 公司/项目名 | 时间段 | 角色
   - 2-4 个 STAR 要点，每个以动词开头，包含量化数据
4. **education**：教育背景，**必须逐字照抄候选人画像中的原文**，不得修改学校名、专业名、日期格式、学历层次等任何信息
5. **additional_highlights**：开源贡献、技术博客、证书、奖项等"""

    HUMAN_TEMPLATE = """请基于以下信息生成优化后的简历。

## 候选人完整画像
{requirement}

## 目标岗位 JD
{extra_info}

## 匹配度评分
{completeness_score}

## 简历格式参考
使用简洁专业的排版风格。每个部分都要有实质内容，不能空着。"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.debug("ResumeOptimizerAgent initialized")

    def invoke(
        self,
        requirement: str = "",
        completeness_score: float = 0.0,
        extra_info: str = "",
    ) -> ResumeResult:
        human_prompt = self.HUMAN_TEMPLATE.format(
            requirement=requirement,
            completeness_score=f"{completeness_score:.2f}",
            extra_info=extra_info or "（无额外信息）",
        )

        logger.info("ResumeOptimizerAgent optimizing")
        return self._call_llm(human_prompt)

    def _call_llm(self, human_prompt: str) -> ResumeResult:
        from datetime import datetime
        now = datetime.now().strftime("%Y 年 %m 月 %d 日")
        messages = [
            SystemMessage(content=f"当前日期: {now}\n\n{self.SYSTEM_PROMPT}"),
            HumanMessage(content=human_prompt),
        ]
        structured_llm = self.llm.with_structured_output(ResumeResult)
        return structured_llm.invoke(messages)
