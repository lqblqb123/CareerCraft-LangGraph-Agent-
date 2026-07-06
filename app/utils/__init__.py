"""File utilities — read PDF resumes, JD files, or plain text input."""

from pathlib import Path

from loguru import logger

# File extensions that can be read as plain text
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".toml"}


def read_input(text_or_path: str) -> str:
    """Smart reader: treats input as file path if the file exists, otherwise as raw text.

    Supports:
    - PDF files → extracted via pdfplumber
    - Text files (.txt, .md, .json, etc.) → read directly
    - Non-file strings → returned as-is (inline text)

    Args:
        text_or_path: Raw text content or a path to a file.

    Returns:
        The extracted/read text content.
    """
    path = Path(text_or_path)

    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return extract_text_from_pdf(text_or_path)
        elif suffix in _TEXT_EXTENSIONS:
            return _read_text_file(text_or_path)
        else:
            # Unknown extension — try reading as text
            logger.warning(f"Unknown file extension '{suffix}', attempting to read as text")
            return _read_text_file(text_or_path)

    # Not a file path — treat as plain text
    return text_or_path


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using pdfplumber.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is unreadable.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "需要安装 pdfplumber 来解析 PDF。\n"
            "运行: pip install pdfplumber"
        )

    text_parts: list[str] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                logger.debug(f"Extracted page {page_num}/{len(pdf.pages)}")
    except Exception as e:
        raise ValueError(f"PDF 解析失败: {e}")

    full_text = "\n\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "PDF 中没有提取到文本内容，可能是扫描版 PDF（图片格式）。"
            "请提供文字版 PDF。"
        )

    logger.info(f"PDF extracted: {len(full_text)} chars from {len(text_parts)} pages")
    return full_text


def _read_text_file(file_path: str) -> str:
    """Read a plain text file (markdown, JSON, etc.)."""
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8")
        logger.info(f"Read text file: {path.name} ({len(content)} chars)")
        return content
    except UnicodeDecodeError:
        # Try GBK encoding (common on Windows for Chinese text)
        try:
            content = path.read_text(encoding="gbk")
            logger.info(f"Read text file (GBK): {path.name} ({len(content)} chars)")
            return content
        except Exception:
            pass
        raise ValueError(f"无法读取文件 {file_path}，请确认文件编码为 UTF-8 或 GBK")
