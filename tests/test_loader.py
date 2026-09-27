from io import BytesIO

import pytest
from PyPDF2 import PdfWriter

from rag.loader import load_document, load_pdf, load_txt


def test_load_txt_utf8():
    assert load_txt("café".encode("utf-8")) == "café"


def test_load_txt_windows_1252_fallback():
    assert load_txt("café".encode("windows-1252")) == "café"


def test_load_document_txt_by_extension():
    assert load_document(b"hello", "notes.TXT") == "hello"


def test_load_document_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(b"not a document", "file.docx")


def test_load_pdf_returns_string():
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    text = load_pdf(buffer.getvalue())
    assert isinstance(text, str)


def test_load_document_routes_pdf_extension():
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    text = load_document(buffer.getvalue(), "manual.pdf")
    assert isinstance(text, str)
