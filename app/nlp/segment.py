"""Деление абзаца на предложения и токены (razdel + пост-правила)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from razdel import sentenize, tokenize

_DASH_START = re.compile(r"^\s*[—–-]\s*[a-zа-яё]")


@dataclass
class SegToken:
    text: str
    start: int  # смещение внутри предложения
    end: int


@dataclass
class SegSentence:
    text: str
    tokens: list[SegToken]
    is_direct_speech: bool


def _merge_author_words(chunks: list[tuple[int, int]], par: str) -> list[tuple[int, int]]:
    """Склеивает реплику и авторские слова: «— Дрыхнешь! — крикнул он.»"""
    out: list[tuple[int, int]] = []
    for start, stop in chunks:
        piece = par[start:stop]
        if out and _DASH_START.match(piece):
            prev = par[out[-1][0] : out[-1][1]].rstrip()
            if prev.endswith(("!", "?", "…", "?»", "!»", "…»")):
                out[-1] = (out[-1][0], stop)
                continue
        out.append((start, stop))
    return out


def split_paragraph(par: str) -> list[SegSentence]:
    chunks = [(s.start, s.stop) for s in sentenize(par)]
    chunks = _merge_author_words(chunks, par)
    result = []
    for start, stop in chunks:
        text = par[start:stop].strip()
        if not text:
            continue
        toks = [SegToken(t.text, t.start, t.stop) for t in tokenize(text)]
        if not toks:
            continue
        is_ds = text.lstrip().startswith(("—", "–", "- ")) or par.lstrip().startswith(("—", "–"))
        result.append(SegSentence(text=text, tokens=toks, is_direct_speech=is_ds))
    return result
