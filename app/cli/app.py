"""Typer CLI application — main command loop and workflow orchestration.

Provides an interactive REPL for the CareerCraft Agent.
Integrates LangGraph workflow with human-in-the-loop interrupts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from langgraph.types import Command
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from app.cli.handlers import CLIHandlers
from app.config.llm import create_llm
from app.graph.state import create_initial_state
from app.graph.workflow import create_workflow
from app.utils import read_input

console = Console()
handlers = CLIHandlers()

cli = typer.Typer(
    name="career-craft",
    help="CareerCraft — 基于 LangGraph 的多 Agent 求职辅导系统",
    add_completion=False,
)


@cli.command()
def main(
    model: str = typer.Option(
        "qwen-plus",
        "--model",
        "-m",
        help="LLM model to use (qwen-plus, qwen-max, etc.)",
    ),
    resume_pdf: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to your resume file (PDF/txt/md)",
    ),
    jd: list[str] = typer.Option(
        None,
        "--jd",
        "-j",
        help="Job description (text or file path). Repeat for multiple JDs: -j a.txt -j b.pdf",
    ),
):
    """Start the CareerCraft interactive CLI.

    Supports multiple JDs: python main.py -r resume.pdf -j jd1.txt -j jd2.md -j "inline JD text"
    """
    _setup_logging()
    handlers.display_welcome()

    try:
        llm = create_llm()
        logger.info(f"LLM initialized: model={model}")
    except ValueError as e:
        console.print(f"[red]❌ LLM 初始化失败: {e}[/red]")
        console.print("[dim]请设置 DASHSCOPE_API_KEY 环境变量[/dim]")
        raise typer.Exit(code=1)

    workflow = create_workflow(llm=llm)

    # If --resume is provided, auto-start with that PDF
    if resume_pdf:
        _start_with_pdf(resume_pdf, jd, workflow)
    elif jd:
        console.print("[dim]使用 --resume 指定简历文件，或直接粘贴简历文本[/dim]")

    # REPL loop
    while True:
        try:
            user_input = _read_command()
            if not user_input:
                continue

            if user_input.startswith("/"):
                should_exit = _handle_command(user_input, workflow)
                if should_exit:
                    break
            else:
                console.print(
                    "[dim]💡 使用 [bold]/new[/bold] 开始新求职辅导会话，"
                    "或直接粘贴简历文本后回车[/dim]"
                )
                _run_career_workflow(user_input, workflow)

        except KeyboardInterrupt:
            console.print("\n[dim]按 Ctrl+C 再次退出，或用 /exit[/dim]")
        except EOFError:
            break

    console.print("[dim]祝求职顺利！[/dim]")


def _start_with_pdf(pdf_path: str, jd_list: list[str] | None, workflow: Any) -> None:
    """Auto-start a career coach session. Supports multiple JDs.

    Args:
        pdf_path: Path to resume file or resume text.
        jd_list: List of JD texts or file paths. Multiple -j flags → multiple JDs.
        workflow: The compiled LangGraph workflow.
    """
    try:
        resume_text = read_input(pdf_path)
        console.print(f"[green]✅ 简历已加载 ({len(resume_text)} 字符)[/green]")

        if jd_list:
            jd_texts = []
            for jd_input in jd_list:
                jd_text = read_input(jd_input)
                jd_texts.append(jd_text)
            console.print(f"[green]✅ {len(jd_list)} 个岗位描述已加载[/green]")
            jd_combined = _combine_jds(jd_texts)
        else:
            jd_combined = "（未提供岗位描述，请先分析简历本身）"

        full_input = f"## 候选人简历\n{resume_text}\n\n{jd_combined}"
        _run_career_workflow(full_input, workflow)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]❌ {e}[/red]")


def _combine_jds(jd_list: list[str]) -> str:
    """Combine multiple JDs into a single structured text for the agent.

    Args:
        jd_list: List of JD text content.

    Returns:
        Formatted combined JD text with clear separators.
    """
    MAX_JDS = 3
    if len(jd_list) == 1:
        return f"## 目标岗位描述\n{jd_list[0]}"

    if len(jd_list) > MAX_JDS:
        skipped = jd_list[MAX_JDS:]
        console.print(
            f"[yellow]⚠️  一次最多分析 {MAX_JDS} 个岗位，前 {MAX_JDS} 个已使用[/yellow]"
        )
        console.print("[dim]跳过的岗位：[/dim]")
        for s in skipped:
            console.print(f"  [dim]- {s[:60]}...[/dim]" if len(s) > 60 else f"  [dim]- {s}[/dim]")
        console.print(f"[dim]完成后重新运行来处理剩余的 {len(skipped)} 个岗位[/dim]")
        jd_list = jd_list[:MAX_JDS]

    parts = ["## 多个目标岗位描述（请逐一分析匹配度并排名）\n"]
    for i, jd in enumerate(jd_list, 1):
        parts.append(f"### 岗位 {i}\n{jd}\n")
    parts.append("请逐一分析候选人简历与每个岗位的匹配度，输出排名和理由。"
                  "自动选择匹配度最高的岗位作为后续优化的目标。")
    return "\n".join(parts)


def _read_command() -> str:
    try:
        return input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _handle_command(cmd: str, workflow: Any) -> bool:
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/exit":
        console.print("[dim]正在退出...[/dim]")
        return True

    elif command == "/help":
        handlers.cmd_help()

    elif command == "/new":
        if not args:
            console.print("[bold]请输入简历内容或 PDF 文件路径：[/bold]")
            console.print("[dim]（拖拽 PDF 文件到终端、粘贴简历文本，或输入多行文本后空行结束）[/dim]")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            args = "\n".join(lines)

        if args.strip():
            _start_new_session(args, workflow)
        else:
            console.print("[yellow]简历内容不能为空[/yellow]")

    elif command == "/jd":
        if args:
            console.print("[green]已记录目标岗位描述[/green]")
            handlers.cmd_save()
        else:
            console.print("[bold]请输入目标岗位描述（JD）：[/bold]")
            console.print("[dim]（粘贴 JD 文本，输入空行结束）[/dim]")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            args = "\n".join(lines)
            if args.strip():
                console.print("[green]已记录目标岗位描述[/green]")

    elif command == "/save":
        handlers.cmd_save()

    elif command == "/load":
        if args:
            state = handlers.cmd_load(args)
            if state:
                console.print("[green]状态已加载，使用 /status 查看[/green]")
        else:
            handlers.cmd_load()

    elif command == "/status":
        handlers.cmd_status()

    elif command == "/resume":
        _resume_workflow(workflow)

    elif command == "/export":
        handlers.cmd_export(args if args else "output")

    elif command == "/re-jd":
        state = handlers.current_state
        if not state:
            console.print("[red]没有活跃的会话，请先 /new 或 /load[/red]")
        else:
            # 从 raw_requirement 中提取简历部分
            raw = state.get("raw_requirement", "")
            resume = raw.split("## 目标岗位描述")[0].replace("## 候选人简历\n", "").strip()
            if not resume:
                resume = raw[:500]  # fallback
            console.print(f"[green]✅ 已复用当前简历 ({len(resume)} 字符)[/green]")
            if not args:
                console.print("[bold]请输入新的目标岗位描述（文件路径或粘贴文本）：[/bold]")
                console.print("[dim]（拖拽文件、粘贴文本，输入空行结束）[/dim]")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
                args = "\n".join(lines)
            if args.strip():
                jd_list = [j.strip() for j in args.split("\n") if j.strip()]
                jd_texts = [read_input(j) for j in jd_list]
                jd_combined = _combine_jds(jd_texts)
                full_input = f"## 候选人简历\n{resume}\n\n{jd_combined}"
                _run_career_workflow(full_input, workflow)

    else:
        console.print(f"[red]未知命令: {command}[/red]")
        console.print("[dim]输入 /help 查看可用命令[/dim]")

    return False


def _start_new_session(input_text: str, workflow: Any) -> None:
    """Start a new career coach session. Auto-detects if input is a file path.

    Args:
        input_text: Resume text, or path to a resume file (PDF/txt/md).
        workflow: The compiled LangGraph workflow.
    """
    # Load resume — supports file path or inline text
    try:
        resume_text = read_input(input_text.strip())
        if resume_text != input_text.strip():
            console.print(f"[green]✅ 简历文件已加载 ({len(resume_text)} 字符)[/green]")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]❌ {e}[/red]")
        return

    console.print()
    console.print("[bold]现在请输入目标岗位描述（JD，支持粘贴文本或文件路径）：[/bold]")
    console.print("[dim]（可以粘贴 JD 文本、拖拽文件路径，输入空行结束。如暂无可留空）[/dim]")
    jd_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        jd_lines.append(line)

    jd_list = [read_input(line.strip()) for line in jd_lines] if jd_lines else ["通用软件开发岗位"]
    jd_combined = _combine_jds(jd_list)

    full_input = f"## 候选人简历\n{resume_text}\n\n{jd_combined}"
    _run_career_workflow(full_input, workflow)


def _run_career_workflow(input_text: str, workflow: Any) -> None:
    """Run the career coach LangGraph workflow with human-in-the-loop.

    Args:
        input_text: Combined resume + JD text.
        workflow: The compiled LangGraph workflow.
    """
    session_id = handlers.cmd_new(input_text)
    state = create_initial_state(input_text)
    handlers.set_current_state(state)

    config = {"configurable": {"thread_id": session_id}}

    console.print()
    console.print("[bold]🔍 开始分析简历与岗位匹配度...[/bold]")
    console.print()

    spinner = Status("[dim]AI 思考中...[/dim]", console=console)

    try:
        is_first_run = True

        while True:
            spinner.start()
            interrupted = False

            if is_first_run:
                stream_input: Any = state
                is_first_run = False
            else:
                stream_input = Command(resume={"answers": state.get("_pending_answer", "")})

            for event in workflow.stream(stream_input, config, stream_mode="updates"):
                spinner.stop()

                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        interrupted = True
                        snapshot = workflow.get_state(config)
                        if snapshot and snapshot.values:
                            state.update(snapshot.values)
                            handlers.set_current_state(state)
                        break

                    if isinstance(node_output, dict):
                        state.update(node_output)
                        handlers.set_current_state(state)
                        _display_node_progress(node_name, node_output)

                if interrupted:
                    break
                spinner.start()

            spinner.stop()

            if not interrupted:
                break

            questions = state.get("questions", [])
            if questions:
                answer = handlers.display_questions(questions)
                state["questions"] = []
                # 不直接改 requirement — 由 human_node 统一处理追加逻辑
                state["_pending_answer"] = answer
                handlers.set_current_state(state)

        console.print()
        console.print(
            Panel.fit(
                "[bold green]🎉 求职辅导完成！[/bold green]\n\n"
                "使用 [bold]/status[/bold] 查看详细状态\n"
                "使用 [bold]/export[/bold] 导出文档（含优化简历、面试准备、求职策略）",
                title="✅ 完成",
            )
        )
        handlers.cmd_save()

    except Exception as e:
        spinner.stop()
        logger.exception(f"Workflow error: {e}")
        msg = str(e)
        if "structured" in msg.lower() or "schema" in msg.lower() or "validation" in msg.lower():
            console.print(f"[red]LLM 输出格式异常（已自动保存状态，可 /resume 重试）: {msg}[/red]")
        else:
            console.print(f"[red]工作流执行出错: {msg}[/red]")
        handlers.cmd_save()


def _resume_workflow(workflow: Any) -> None:
    if not handlers.current_session_id:
        console.print("[red]没有活跃的会话。请先 /load <id> 或 /new[/red]")
        return

    state = handlers.current_state
    if not state:
        state = handlers.repository.load_state(handlers.current_session_id)

    if not state:
        console.print("[red]无法加载会话状态[/red]")
        return

    requirement = state.get("raw_requirement", "") or state.get("requirement", "")
    if not requirement:
        console.print("[red]会话中没有找到简历信息[/red]")
        return

    console.print(
        f"[green]恢复会话 {handlers.current_session_id[:8]}... "
        f"当前状态: {state.get('status', 'UNKNOWN')}[/green]"
    )
    _run_career_workflow(requirement, workflow)


def _display_node_progress(node_name: str, output: dict[str, Any]) -> None:
    status = output.get("status", "")

    icons = {
        "requirement_node": "🔍",
        "architect_node": "📝",
        "review_node": "🔎",
        "prompt_node": "📋",
        "export_node": "📄",
    }

    messages = {
        "ADVISOR_ANALYZED": "简历分析完成",
        "HUMAN_INPUT_RECEIVED": "收到候选人补充",
        "RESUME_OPTIMIZED": "简历优化完成",
        "REVIEWED": "审查完成",
        "STRATEGY_GENERATED": "求职策略完成",
        "COMPLETED": "工作流完成",
    }

    icon = icons.get(node_name, "⚙️")
    msg = messages.get(status, f"{node_name} 完成")

    if node_name == "requirement_node":
        score = output.get("completeness_score", 0)
        is_complete = output.get("is_complete", False)
        questions = output.get("questions", [])
        # Check requirement text for multi-JD ranking
        req_text = output.get("requirement", "")
        jd_count = req_text.count("匹配度:") if req_text else 0

        status_text = f"  {icon} {msg} — 匹配度: [cyan]{score:.0%}[/cyan]"
        if jd_count > 1:
            status_text += f" | [dim]{jd_count} 个 JD 已排名[/dim]"
        status_text += " " + ("[green]✓ 已完整[/green]" if is_complete else f"[yellow]待补充 ({len(questions)} 个问题)[/yellow]")
        console.print(status_text)
    elif node_name == "review_node":
        console.print(f"  {icon} {msg}")
    else:
        console.print(f"  {icon} {msg}")


def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        format="<level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
    )
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        logger.add(
            "logs/career-coach.log",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
    except Exception:
        pass


if __name__ == "__main__":
    cli()
