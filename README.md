# CareerCraft

基于 LangGraph 的多 Agent 求职辅导系统。上传简历 PDF + 岗位描述，4 个协作 Agent 自动完成：匹配度评分 → 追问澄清 → 简历优化 → 审查质检 → 能力提升建议。

## 架构

```
START → 求职顾问（匹配评分+追问）⇄ 你回答（最多3轮）
    → 简历优化师（STAR重写）
    → 简历审查员（真实性/ATS检查，输出风险提示）
    → 求职策略（能力差距分析）
    → 导出 Markdown + PDF
```

4 个 Agent 通过 LangGraph StateGraph 编排，Pydantic 约束结构化输出，SQLite 持久化会话状态。

## 快速开始

```bash
# 安装
pip install -e .

# 设置 API Key
set DASHSCOPE_API_KEY=your_key

# 交互模式
python main.py

# 直接指定简历 + JD
python main.py -r 简历.pdf -j JD.txt

# 多 JD 匹配排名（最多 3 个）
python main.py -r 简历.pdf -j JD1.txt -j JD2.txt -j JD3.txt
```

## 命令

| 命令 | 功能 |
|------|------|
| `/new` | 开始新会话（支持 PDF/txt/md 文件或粘贴文本） |
| `/status` | 查看当前进度 |
| `/export` | 导出结果到 output/ 目录 |
| `/save` / `/load` | 保存/恢复会话 |
| `/re-jd` | 换一批 JD 重新分析（复用已上传简历） |

## 输出

```
output/
├── resume.md          # 优化后的简历
├── growth_plan.md     # 能力差距分析与提升建议
├── full_report.md     # 完整报告（含匹配排名+风险提示）
└── resume.pdf         # 简历 PDF
```

## 匹配度公式

```
match_score = 技能 × 0.40 + 经验 × 0.35 + 行业 × 0.15 + 学历软技 × 0.10
```

每项 0-100 分，LLM 按标准评分后加权计算，评分过程在报告中可见。

## 技术栈

Python 3.12 · LangGraph · LangChain · Pydantic v2 · Typer · Rich · SQLite · Jinja2 · pdfplumber · fpdf2

## 项目结构

```
app/
├── agents/            # 4 个 Agent（Prompt + 结构化输出）
│   ├── requirement.py # 求职顾问
│   ├── architect.py   # 简历优化师
│   ├── reviewer.py    # 简历审查员
│   └── prompt.py      # 求职策略
├── graph/             # LangGraph 工作流
│   ├── state.py       # 中央状态定义
│   ├── nodes.py       # 节点函数
│   ├── router.py      # 条件路由
│   └── workflow.py    # 图编译
├── cli/               # Typer 命令行
├── exporter/          # Markdown + PDF 导出
├── storage/           # SQLite 持久化
├── config/            # LLM 配置
└── utils/             # PDF 解析等工具
tests/
├── test_workflow.py   # 工作流单元测试
├── test_eval.py       # 匹配度评测脚本
└── eval_cases.json    # 评测数据集（20 组）
```
