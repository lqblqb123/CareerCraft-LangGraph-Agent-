"""Command handlers for the CareerCraft CLI.

Each handler corresponds to a CLI command (e.g., /new, /save, /load, /export).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.exporter.markdown import MarkdownExporter
from app.storage.repository import SessionRepository

console = Console()


class CLIHandlers:
    """Handles the execution of CLI commands."""

    def __init__(self, repository: SessionRepository | None = None):
        self.repository = repository or SessionRepository()
        self.exporter = MarkdownExporter()
        self._current_session_id: str | None = None
        self._current_state: dict[str, Any] | None = None

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @property
    def current_state(self) -> dict[str, Any] | None:
        return self._current_state

    def set_current_state(self, state: dict[str, Any]) -> None:
        self._current_state = state

    def cmd_new(self, input_text: str) -> str:
        """Handle /new command — start a new career coach session.

        Args:
            input_text: Combined resume and JD text.

        Returns:
            The session ID.
        """
        title = input_text[:50] + "..." if len(input_text) > 50 else input_text
        session = self.repository.create_session(
            title=title,
            raw_requirement=input_text,
        )
        self._current_session_id = session.id
        logger.info(f"New session created: {session.id}")

        console.print(
            Panel.fit(
                f"[bold green]新求职辅导会话已创建[/bold green]\n"
                f"Session ID: [cyan]{session.id}[/cyan]",
                title="📋 /new",
            )
        )
        return session.id

    def cmd_save(self, state: dict[str, Any] | None = None) -> None:
        if not self._current_session_id:
            console.print("[red]没有活跃的会话，请先使用 /new[/red]")
            return

        state_to_save = state or self._current_state
        if not state_to_save:
            console.print("[yellow]没有可保存的状态[/yellow]")
            return

        self.repository.save_state(self._current_session_id, state_to_save)
        console.print(
            f"[green]✅ 状态已保存到 Session {self._current_session_id[:8]}...[/green]"
        )

    def cmd_load(self, session_id: str | None = None) -> dict[str, Any] | None:
        if session_id is None:
            self._list_sessions()
            return None

        session = self.repository.get_session(session_id)
        if not session:
            console.print(f"[red]未找到 Session: {session_id}[/red]")
            return None

        state = self.repository.load_state(session_id)
        self._current_session_id = session_id
        self._current_state = state

        console.print(
            Panel.fit(
                f"[bold green]会话已加载[/bold green]\n"
                f"标题: {session.title}\n"
                f"状态: {session.status}\n"
                f"匹配度: {session.completeness_score:.0%}",
                title="📂 /load",
            )
        )
        return state

    def cmd_status(self) -> None:
        if not self._current_state:
            if self._current_session_id:
                state = self.repository.load_state(self._current_session_id)
                self._current_state = state
            else:
                console.print("[yellow]没有活跃的会话[/yellow]")
                return

        state = self._current_state
        if not state:
            console.print("[yellow]没有状态信息[/yellow]")
            return

        status = state.get("status", "UNKNOWN")
        match_score = state.get("completeness_score", 0.0)
        table = Table(title="📊 求职辅导状态")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="white")

        table.add_row("状态", status)
        table.add_row("简历匹配度", f"{match_score:.0%}")

        arch = state.get("architecture", {})
        if arch:
            table.add_row("简历优化", f"已完成 ({len(arch)} 个模块)")

        if state.get("growth_plan", ""):
            table.add_row("求职策略", "已完成")

        console.print(table)

    def cmd_export(self, output_dir: str = "output") -> None:
        if not self._current_state:
            if self._current_session_id:
                self._current_state = self.repository.load_state(
                    self._current_session_id
                )
            if not self._current_state:
                console.print("[red]没有可导出的内容[/red]")
                return

        state = self._current_state

        arch = state.get("architecture", {})
        if not arch:
            console.print("[yellow]简历尚未完成优化，请等待工作流完成[/yellow]")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exporter = MarkdownExporter()
        exported_files = exporter.export_all(state, str(output_path))

        console.print(
            Panel.fit(
                "\n".join(
                    f"[green]✅ {f}[/green]" for f in exported_files
                ),
                title="📄 /export — 文件已导出",
            )
        )

    def cmd_help(self) -> None:
        help_text = """
[bold cyan]可用命令[/bold cyan]

[bold]/new[/bold]      — 开始新的求职辅导会话（支持 PDF 路径或粘贴简历）
[bold]/resume[/bold]   — 继续上次中断的会话（需先 /load）
[bold]/save[/bold]     — 保存当前状态
[bold]/load[/bold]     — 加载之前的会话（/load <id> 加载指定会话）
[bold]/status[/bold]   — 查看当前辅导进度
[bold]/export[/bold]   — 导出生成文档到 output/ 目录
[bold]/re-jd[/bold]    — 换一批 JD 重新分析（复用当前简历）
[bold]/help[/bold]     — 显示此帮助信息
[bold]/exit[/bold]     — 退出程序

[bold cyan]使用示例[/bold cyan]

  /new
  请输入简历内容或 PDF 文件路径: C:\\Users\\me\\简历.pdf

  或直接粘贴简历文本后输入空行提交。

  在辅导澄清阶段，直接输入你的回答即可。

[bold cyan]快速启动[/bold cyan]

  python main.py -r 简历.pdf -j "岗位名称: 后端开发\n要求: ..."
"""
        console.print(Panel.fit(help_text, title="❓ /help"))

    def _list_sessions(self) -> None:
        sessions = self.repository.list_sessions(limit=10)
        if not sessions:
            console.print("[yellow]没有历史会话[/yellow]")
            return

        table = Table(title="📂 历史会话")
        table.add_column("ID", style="cyan")
        table.add_column("标题", style="white")
        table.add_column("状态", style="yellow")
        table.add_column("更新时间", style="dim")

        for s in sessions:
            table.add_row(
                s.id,
                s.title[:40],
                s.status,
                s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else "",
            )

        console.print(table)
        console.print(
            "[dim]使用 /load <id> 加载指定会话[/dim]"
        )

    def display_questions(self, questions: list[str]) -> str:
        """Display clarification questions and collect candidate answers.

        Args:
            questions: List of questions from the Career Advisor Agent.

        Returns:
            The candidate's combined answers.
        """
        console.print()
        console.print(
            Panel.fit(
                "[bold yellow]⚠️ 需要补充以下经历细节，让简历更有竞争力：[/bold yellow]",
                title="🔍 经历挖掘",
            )
        )

        for i, q in enumerate(questions, 1):
            console.print(f"  [cyan]{i}.[/cyan] {q}")

        console.print()
        console.print("[dim]请输入补充信息（可以分段回答）：[/dim]")
        console.print("[dim]（输入空行结束。如果确实不清楚，可以回答「不确定」来跳过）[/dim]")

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            except EOFError:
                break

        answer = "\n".join(lines) if lines else ""
        if not answer:
            answer = "此项不确定，请 AI 给出默认建议"

        logger.info(f"Received candidate answer: {len(answer)} chars")
        return answer

    def display_welcome(self) -> None:
        """Display the welcome banner."""
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]🎯 CareerCraft[/bold cyan]\n"
                "[dim]基于 LangGraph 的多 Agent 求职辅导系统[/dim]\n\n"
                "简历 PDF → 差距分析 → 追问经历 → 简历优化 → 求职策略\n\n"
                "输入 [bold]/help[/bold] 查看命令，[bold]/new[/bold] 开始求职辅导",
                title="🏆 欢迎",
            )
        )
        console.print()
