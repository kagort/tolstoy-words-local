"""Короткие подписи для графиков: «Лев Николаевич Толстой» → «Л. Н. Толстой», длинные названия — с «…».

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
    """Инициалы и фамилия: «Лев Николаевич Толстой» и «Толстой Л. Н.» → «Л. Н. Толстой»; соавторы — каждый;
    псевдонимы из одного слова и не-имена («Представитель США в ООН») — как есть; пометка в скобках
    («(предположительно)») сокращается до «(?)», полная подпись — в подсказке и выгрузках."""
    if not author:
        return ""
    author = author.strip()
    if "," in author:
        return clip(", ".join(short_author(a) for a in author.split(",") if a.strip()), AUTHOR_MAX)
    note = ""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", author)
    if m and m.group(1):
        author, note = m.group(1).strip(), (" (?)" if "предполож" in m.group(2).lower() else "")
    words = [w for w in re.split(r"\s+", author) if w]
    if len(words) < 2 or not all(w[:1].isupper() for w in words):
        return clip(author + note, AUTHOR_MAX)
    initials = [w for w in words if re.fullmatch(r"[A-ZА-ЯЁ]\.(?:[A-ZА-ЯЁ]\.)*", w)]
    if initials:  # уже с инициалами: «Толстой Л. Н.», «Л. Н. Толстой», «Л.Н. Толстой»
        rest = [w for w in words if w not in initials]
        if len(rest) != 1:
            return clip(author + note, AUTHOR_MAX)
        letters = [ch + "." for w in initials for ch in w.replace(".", "")]
        return clip(" ".join(letters) + " " + rest[0] + note, AUTHOR_MAX)
    *names, last = words
    return clip(" ".join(f"{n[0]}." for n in names) + f" {last}" + note, AUTHOR_MAX)


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
