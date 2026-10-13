"""Извлечение текста из файлов и нормализация."""

from __future__ import annotations

import html
import io
import re
import unicodedata
import zipfile
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = (".txt", ".docx", ".fb2")


class UnsupportedFormat(ValueError):
    pass


def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml = zf.read("word/document.xml")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ElementTree.fromstring(xml)
    paras = []
    for p in root.iter(f"{{{ns['w']}}}p"):
        parts = [t.text or "" for t in p.iter(f"{{{ns['w']}}}t")]
        paras.append("".join(parts))
    return "\n".join(paras)


def _fb2_text(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    paras = []
    for body in root.iter():
        if body.tag.endswith("body"):
            for el in body.iter():
                if el.tag.endswith("}p") or el.tag == "p":
                    paras.append("".join(el.itertext()))
    return "\n".join(paras)


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return decode_bytes(data)
    if name.endswith(".docx"):
        return _docx_text(data)
    if name.endswith(".fb2"):
        return _fb2_text(data)
    raise UnsupportedFormat(f"Неподдерживаемый формат файла: {filename}. Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}")


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t   ]+")


def normalize_text(text: str) -> str:
    """Unicode NFC, HTML-сущности, переносы, лишние пробелы. Абзацы = непустые строки."""
    text = text.replace("\x00", "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # склейка переноса слова: «обо-\nняние» → «обоняние» (только если вторая часть со строчной)
    text = _HYPHEN_BREAK.sub(lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), text)
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n") if p.strip()]
