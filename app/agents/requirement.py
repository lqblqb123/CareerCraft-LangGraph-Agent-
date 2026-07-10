"""求职顾问 Agent — 分析简历、识别薄弱项、追问经历细节。

这是求职辅导工作流的第一步，负责将简历原文和目标岗位转化为
结构化的候选人画像和待澄清问题。

支持单 JD 和多 JD 模式：多 JD 时自动排名并选最优。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class JdMatch(BaseModel):
    """单个 JD 的匹配结果。"""

    jd_index: int = Field(description="岗位序号（从1开始）")
    jd_summary: str = Field(default="", description="岗位名称，必须照抄 JD 原文，禁止改写")
    match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="匹配度")


class CareerAdvisorResult(BaseModel):
    """求职顾问的结构化输出 —— 所有字段的读写关系见 state.py。"""

    # ↓ 写入 state["requirement"]，后续 architect/reviewer/prompt 节点都读它
    candidate_profile: str = Field(
        default="",
        description="候选人画像：技能、经验、行业背景的结构化总结",
    )

    # ↓ 写入 state["missing_info"]，architect_node 读，json.dumps 后拼进优化师的 extra_info
    gaps: list[dict[str, str]] = Field(
        default_factory=list,
        description="与最优岗位的差距，每项包含 category、gap、importance",
    )

    # ↓ 写入 state["completeness_score"]，router 判断是否继续追问 + report 导出
    match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="与最优岗位的综合匹配度（0.0-1.0），由 AI 根据技术栈、经验、行业、学历等综合判断",
    )

    # ↓ 拼成排名文本追加到 state["requirement"]，最终显示在 full_report 第一章，单 JD 时为空
    match_ranking: list[JdMatch] = Field(
        default_factory=list,
        description="多 JD 排名（单 JD 时为空），按匹配度从高到低排列",
    )

    # ↓ 只在日志打印，不写入 state（仅用于问题排查）
    best_jd_index: int = Field(
        default=0,
        description="最优岗位序号（1-based），仅用于 logger，不写入 state",
    )

    # ↓ 写入 state["questions"]，router 判断是否有待问问题 + CLI 展示给用户
    questions: list[str] = Field(
        default_factory=list,
        description="待追问的问题（每轮 1-2 个、20 字以内、只问做过没、不问怎么做的）",
    )

    # ↓ 写入 state["is_complete"]，router 判断是否继续追问循环
    is_complete: bool = Field(
        default=False,
        description="信息是否足够完整（match_score >= 0.6 且无 critical 缺失）",
    )


class CareerAdvisorAgent:
    """求职顾问 Agent —— 分析简历和 JD，识别差距，追问经历细节。

    核心原则：
    - 绝对禁止编造经历
    - 多 JD 时自动排名，选最优进入后续流程
    - 量化驱动：每条经历都应该有数字支撑
    """

    name = "CareerAdvisorAgent"

    SYSTEM_PROMPT = """你是一个资深的职业规划顾问和求职导师。你会收到候选人的简历和一个或多个岗位描述（JD），你需要系统性地分析匹配度并挖掘隐藏亮点。

## 你的职责
1. 解析简历内容，提取候选人的技能、经验和核心优势
2. 对比目标 JD（如有多份则逐一对比），识别匹配点和差距
3. 发现简历中"写了但没写清楚"或"做了但没写"的经历
4. 生成精准的追问问题，帮助候选人补充关键信息
5. 评估简历的面试竞争力

## 多 JD 处理规则（重要）
当收到多个岗位描述时：
- **逐一分析**每个岗位与简历的匹配度，不要跳过任何一个
- **客观排名**：按匹配度从高到低排列，输出到 match_ranking 字段
- **自动选择最优**：best_jd_index 设为匹配度最高的岗位序号
- 后续的 gaps、questions 基于最优岗位生成
- 如果两个岗位匹配度接近（差距 < 0.1），在 candidate_profile 中注明
- **严禁跳过某个 JD**：即使明显不匹配，也要给出分析结果
- **jd_summary 只能照抄原文**：jd_summary 必须直接复制 JD 原标题或开头第一句话，严禁自己概括、改写或添加任何 JD 中没有的词汇（比如 JD 写"后端开发"，你不能写成"后端开发工程师（高并发/微服务）"）

## 核心原则
- **绝对禁止编造**：任何不确定的信息都必须向候选人确认，严禁猜测或编造经历
- **用户授权即默认**：当候选人对某个维度的问题回复"不知道"、"随便"、"不懂"、"你定"、"不清楚"等不表态内容时，视为候选人仅就该维度授权 AI 自行决策。该维度标记为已处理、不再追问，其他未解决的问题仍需继续提问。仅当所有关键维度都已澄清后，is_complete 才为 true
- **量化驱动**：持续追问"做到了什么程度？能不能用数字衡量？"

## 匹配度评估

根据候选人与目标岗位的综合匹配程度给出 0.0-1.0 的评分，不需要计算具体公式，凭你的专业判断给出最合理的分数。

## 追问策略
- **严禁重复**：**绝对不允许**重复提问之前已经问过的问题
- **极致精简**：每轮最多 1-2 个问题，每个问题控制在 20 字以内。**一问只问一件事**，严禁一个问句里塞多个问题（反面例子："是否独立完成？有无产线部署？"——这是两个问题，必须拆开各问各的）
- **不要客套**：不用"请问""能不能说一下"等礼貌前缀，直接问核心
- **只问缺口**：已写在简历里的不问、候选人不负责的不问、JD 没要求的不问、不问具体项目的细节案例
- **禁止索要输入**：简历和 JD 已在输入中，严禁追问"请提供简历/JD/岗位描述"等索取输入文件的问题
- **只问做过没**：问的是"有没有做过、会不会"，不问"怎么做的、为什么这么做"。好的问题："用过 K8s 吗？""带过团队吗？""接触过高并发吗？"——全是能用是/否或一个数字回答的。不要问"K8s 怎么部署的""描述一下你的系统架构"这类需要展开解释的问题
- 第 3 轮后将强制进入优化阶段
- 参考格式：\""模板：你负责了支付模块哪部分？独立还是协作？"/"QPS 优化前后数据？"/"有没有技术博客或开源项目？\""""

    HUMAN_TEMPLATE = """请分析以下简历和岗位需求，识别信息缺口。

## 候选人简历
{raw_requirement}

## 岗位描述
{requirement}

## 当前已分析轮次
{rounds_count} 轮

## 之前已问过的问题（严禁重复提问）
{previous_questions}

请分析并输出JSON结果。如果是多 JD，请逐一排名。如果简历信息已经足够完整（match_score >= 0.6 且无 critical 缺失），将 questions 设为空数组，is_complete 设为 true。"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        logger.debug("CareerAdvisorAgent initialized")

    def invoke(
        self,
        raw_requirement: str = "",
        requirement: str = "",
        rounds_count: int = 0,
        previous_questions: str = "",
    ) -> CareerAdvisorResult:
        """分析简历和JD，识别信息缺口。支持多JD排名。"""
        human_prompt = self.HUMAN_TEMPLATE.format(
            raw_requirement=raw_requirement,
            requirement=requirement or raw_requirement,
            rounds_count=rounds_count,
            previous_questions=previous_questions or "（首次分析，暂无历史问题）",
        )

        logger.info(f"CareerAdvisorAgent analyzing (round {rounds_count + 1})")
        result = self._call_llm(human_prompt)

        jd_count = len(result.match_ranking)
        logger.info(
            f"CareerAdvisorAgent: best_jd=#{result.best_jd_index}, "
            f"match_score={result.match_score:.2f}, "
            f"is_complete={result.is_complete}, "
            f"jds_ranked={jd_count}, "
            f"questions={len(result.questions)}"
        )
        return result

    def _call_llm(self, human_prompt: str) -> CareerAdvisorResult:
        from datetime import datetime
        now = datetime.now().strftime("%Y 年 %m 月 %d 日")
        messages = [
            SystemMessage(content=f"当前日期: {now}\n\n{self.SYSTEM_PROMPT}"),
            HumanMessage(content=human_prompt),
        ]
        structured_llm = self.llm.with_structured_output(CareerAdvisorResult)
        return structured_llm.invoke(messages)
