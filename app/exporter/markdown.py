"""Markdown exporter — renders Jinja2 templates to produce career coach output files.

Generates: resume.md, growth_plan.md, full_report.md, resume.pdf
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger


class MarkdownExporter:
    """Exports workflow state to Markdown files using Jinja2 templates."""

    TEMPLATES = {
        "resume.md": "resume.md.j2",
        "growth_plan.md": "growth_plan.md.j2",
        "full_report.md": "full_report.md.j2",
    }

    def __init__(self, template_dir: str | None = None):
        if template_dir is None:
            template_dir = str(
                Path(__file__).resolve().parent.parent / "templates"
            )
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.debug(f"MarkdownExporter initialized with templates from {template_dir}")

    def export_all(self, state: dict[str, Any], output_dir: str) -> list[str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        context = self._prepare_context(state)
        exported = []

        for filename, template_name in self.TEMPLATES.items():
            try:
                content = self._render(template_name, context)
                file_path = output_path / filename
                file_path.write_text(content, encoding="utf-8")
                exported.append(str(file_path))
                logger.info(f"Exported: {file_path}")
            except Exception as e:
                logger.error(f"Failed to export {filename}: {e}")

        # Also export PDF version of resume
        try:
            from app.exporter.pdf import export_resume_pdf
            pdf_path = output_path / "resume.pdf"
            export_resume_pdf(state, str(pdf_path))
            exported.append(str(pdf_path))
            logger.info(f"Exported PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")

        return exported

    def _prepare_context(self, state: dict[str, Any]) -> dict[str, Any]:
        arch = state.get("architecture", {})

        return {
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "match_score": state.get("completeness_score", 0.0),
            "candidate_profile": state.get("requirement", ""),
            "personal_summary": arch.get("personal_summary", ""),
            "skills_matrix": arch.get("skills_matrix", ""),
            "work_experience": arch.get("work_experience", ""),
            "additional_highlights": arch.get("additional_highlights", ""),
            "review_feedback": state.get("review_feedback", ""),
            "hard_data_warnings": arch.get("hard_data_warnings", ""),
            "growth_plan": state.get("growth_plan", ""),
        }

    def _render(self, template_name: str, context: dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)
