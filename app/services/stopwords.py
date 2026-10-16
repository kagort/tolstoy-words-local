"""Стоп-слова: встроенный список для русского и свои списки пользователя.

Стоп-слова — параметр просмотра: данные в базе не меняются, слова лишь скрываются из списков
(частотные леммы портрета текста, связи в профиле слова и отчёте). Значение параметра `stop`:
"" — не исключать, "ru" — встроенный список, число — id своего списка.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import StopList
from app.nlp.tree import norm

BUILTIN = "ru"
BUILTIN_NAME = "Встроенный (русский)"
BUILTIN_FILE = Path(__file__).resolve().parent.parent / "data" / "stopwords_ru.txt"


def parse_words(text: str) -> list[str]:
    """Слова из текста: через пробел, запятую, точку с запятой или с новой строки; # — комментарий."""
    out, seen = [], set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for w in line.replace(",", " ").replace(";", " ").split():
            w = norm(w.strip())
            if w and w not in seen:
                seen.add(w)
                out.append(w)
    return out


@lru_cache(maxsize=1)
def builtin_words() -> tuple[str, ...]:
    return tuple(parse_words(BUILTIN_FILE.read_text(encoding="utf-8")))


def options(session: Session) -> list[dict]:
    """Варианты для выпадающего списка «Стоп-слова»."""
    opts = [{"value": "", "label": "не исключать"},
            {"value": BUILTIN, "label": f"{BUILTIN_NAME} — {len(builtin_words())} слов"}]
    for sl in session.scalars(select(StopList).order_by(StopList.name)):
        opts.append({"value": str(sl.id), "label": f"{sl.name} — {len(sl.words or [])} слов"})
    return opts


def resolve(session: Session, value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    if value == BUILTIN:
        return frozenset(builtin_words())
    if value.isdigit():
        sl = session.get(StopList, int(value))
        if sl is not None:
            return frozenset(norm(w) for w in sl.words or [])
    return frozenset()


def label(session: Session, value: str | None) -> str:
    return next((o["label"] for o in options(session) if o["value"] == (value or "")), "не исключать")
