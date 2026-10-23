"""Короткие подписи для графиков: «Лев Николаевич Толстой» → «Толстой Л. Н.», длинные названия — с «…».

Полная подпись остаётся во всплывающей подсказке и в выгрузках; у текста можно задать своё короткое
название (meta["short_title"], колонка «короткое» в таблице метаданных).
"""

from __future__ import annotations

import re

TITLE_MAX = 28
AUTHOR_MAX = 24


def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip(" ,.;:—-") + "…"


def short_author(author: str | None) -> str:
    """«Имя Отчество Фамилия» → «Фамилия И. О.»; «Фамилия И.» и псевдонимы из одного слова — как есть; соавторы — каждый."""
    if not author:
        return ""
    if "," in author:
        return clip(", ".join(short_author(a) for a in author.split(",") if a.strip()), AUTHOR_MAX)
    words = [w for w in re.split(r"\s+", author.strip()) if w]
    if len(words) < 2 or any(w.endswith(".") for w in words) or not all(w[:1].isupper() for w in words):
        return clip(author.strip(), AUTHOR_MAX)
    *names, last = words
    return clip(f"{last} " + " ".join(f"{n[0]}." for n in names), AUTHOR_MAX)


def short_title(title: str | None, meta: dict | None = None) -> str:
    own = ((meta or {}).get("short_title") or "").strip()
    return own or clip(title or "", TITLE_MAX)


def text_labels(title: str, author: str | None, year: int | None, meta: dict | None = None) -> tuple[str, str]:
    """(короткая, полная) подпись текста."""
    full = (f"{author}, " if author else "") + f"«{title}»" + (f" ({year})" if year else "")
    a = short_author(author)
    short = (f"{a} · " if a else "") + short_title(title, meta)
    return short, full


def unique(labels: list[str]) -> list[str]:
    """Одинаковые короткие подписи на оси Plotly слились бы в одну категорию — добавляем номер."""
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        n = seen.get(lab, 0) + 1
        seen[lab] = n
        out.append(lab if n == 1 else f"{lab} ({n})")
    return out
