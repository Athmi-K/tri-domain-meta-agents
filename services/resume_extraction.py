"""
app/services/resume_extraction.py

Extracts plain text from an uploaded resume file, entirely in memory -
no file is ever written to disk. Supports PDF, DOCX, and plain text
formats. This is the piece that was missing: the frontend currently
never sends real file content for PDF/DOCX, only a filename string, so
nothing downstream (resume_optimizer, career_agent.run) ever had actual
resume text to work with.

Install deps:
    pip install pdfplumber python-docx --break-system-packages
"""

from __future__ import annotations

import io
from pathlib import Path

import pdfplumber
from docx import Document


class ResumeExtractionError(Exception):
    """Raised when a resume file can't be parsed into text."""


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Sanity limits - a resume shouldn't be huge; this guards against
# accidental huge uploads chewing up memory/CPU during extraction.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from resume file bytes already held in memory.

    Raises ResumeExtractionError with a user-facing message on any
    failure (unsupported type, corrupt file, empty result) so the API
    layer can return a clean 400 instead of a raw 500.
    """
    if not file_bytes:
        raise ResumeExtractionError("Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ResumeExtractionError(
            f"File is too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"Max size is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ResumeExtractionError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if suffix == ".pdf":
            text = _extract_pdf_text(file_bytes)
        elif suffix == ".docx":
            text = _extract_docx_text(file_bytes)
        else:  # .txt / .md
            text = _extract_plain_text(file_bytes)
    except ResumeExtractionError:
        raise
    except Exception as exc:
        raise ResumeExtractionError(
            f"Could not read the resume file: {exc}"
        ) from exc

    text = text.strip()

    if not text:
        raise ResumeExtractionError(
            "No readable text found in the resume. If this is a scanned "
            "image-based PDF, text extraction won't work - try a "
            "text-based export instead."
        )

    return text


def _extract_pdf_text(file_bytes: bytes) -> str:
    text_parts: list[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def _extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(
        paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()
    )


def _extract_plain_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")