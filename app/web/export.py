"""Local conversation export helpers for the VOCA web workspace."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import re

from app.conversations.store import Conversation, StoredMessage


def export_conversation(
    conversation: Conversation, messages: list[StoredMessage], export_format: str
) -> tuple[bytes, str, str]:
    """Return a downloadable TXT or PDF representation of one local conversation."""
    if export_format not in {"txt", "pdf"}:
        raise ValueError("Export format must be 'txt' or 'pdf'.")

    filename_base = _safe_filename(conversation.title)
    if export_format == "txt":
        return _text_export(conversation, messages).encode("utf-8"), "text/plain; charset=utf-8", f"{filename_base}.txt"
    return _pdf_export(conversation, messages), "application/pdf", f"{filename_base}.pdf"


def _text_export(conversation: Conversation, messages: list[StoredMessage]) -> str:
    lines = [
        conversation.title,
        "=" * len(conversation.title),
        f"Exported from VOCA on {datetime.now().strftime('%d %b %Y, %H:%M')}",
        "Private local conversation export",
        "",
    ]
    for message in messages:
        speaker = "You" if message.role == "user" else "VOCA"
        lines.extend([f"{speaker}:", message.content, ""])
    return "\n".join(lines)


def _pdf_export(conversation: Conversation, messages: list[StoredMessage]) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as error:
        raise RuntimeError("PDF export needs ReportLab. Run: pip install -r requirements.txt") from error

    font_name = "Helvetica"
    windows_font = Path(r"C:\Windows\Fonts\arial.ttf")
    if windows_font.is_file():
        font_name = "VOCAArial"
        try:
            pdfmetrics.registerFont(TTFont(font_name, str(windows_font)))
        except Exception:
            font_name = "Helvetica"

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title=conversation.title,
        author="VOCA",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VocaTitle", parent=styles["Title"], fontName=font_name, fontSize=20, leading=25, textColor=colors.HexColor("#14213D"), alignment=TA_LEFT
    )
    meta_style = ParagraphStyle(
        "VocaMeta", parent=styles["Normal"], fontName=font_name, fontSize=9, leading=13, textColor=colors.HexColor("#5B6B82")
    )
    user_style = ParagraphStyle(
        "VocaUser", parent=styles["Normal"], fontName=font_name, fontSize=11, leading=16, textColor=colors.HexColor("#203A70"), spaceAfter=4
    )
    assistant_style = ParagraphStyle(
        "VocaAssistant", parent=styles["Normal"], fontName=font_name, fontSize=11, leading=16, textColor=colors.HexColor("#18243A"), spaceAfter=4
    )
    speaker_style = ParagraphStyle(
        "VocaSpeaker", parent=styles["Normal"], fontName=font_name, fontSize=9, leading=12, textColor=colors.HexColor("#148778"), spaceBefore=10, spaceAfter=3
    )
    story = [
        Paragraph(escape(conversation.title), title_style),
        Spacer(1, 4 * mm),
        Paragraph(f"Exported from VOCA on {datetime.now().strftime('%d %b %Y, %H:%M')}", meta_style),
        Paragraph("Private local conversation export", meta_style),
        Spacer(1, 5 * mm),
    ]
    for message in messages:
        speaker = "YOU" if message.role == "user" else "VOCA"
        body_style = user_style if message.role == "user" else assistant_style
        body = escape(message.content).replace("\n", "<br/>")
        story.extend([Paragraph(speaker, speaker_style), Paragraph(body, body_style)])

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return output.getvalue()


def _page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColorRGB(0.82, 0.86, 0.92)
    canvas.line(document.leftMargin, 13 * 25.4 / 72, document.pagesize[0] - document.rightMargin, 13 * 25.4 / 72)
    canvas.setFillColorRGB(0.35, 0.42, 0.51)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(document.leftMargin, 9 * 25.4 / 72, "VOCA - private local export")
    canvas.drawRightString(document.pagesize[0] - document.rightMargin, 9 * 25.4 / 72, f"Page {document.page}")
    canvas.restoreState()


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", title).strip().replace(" ", "-")
    return cleaned[:60] or "voca-conversation"
