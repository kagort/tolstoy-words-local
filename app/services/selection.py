"""Единая «Выборка» текстов и слов для всех вкладок анализа и экспорта.

Параметры запроса (одинаковые везде):
    corpus_id                  — корпус (пусто — все готовые тексты);
    texts=…&texts=…            — отмеченные тексты внутри корпуса (пусто — все);
    authors, genres, stypes    — фильтры по сведениям о текстах (пусто — без фильтра);
    year_from, year_to         — диапазон лет (тексты без года при этом фильтре исключаются);
    words=…&words=…            — выбранные слова словаря (id статей);
    stop                       — стоп-слова: "" — нет, "ru" — встроенный список, число — свой список;
    noprop=1                   — без имён собственных (TF-IDF, ключевые слова, темы, семантика, коллокации, стилометрия).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TextDoc
from app.services import analytics as A

SELECTION_KEYS = ("corpus_id", "texts", "authors", "genres", "stypes", "year_from", "year_to")
LIST_KEYS = {"texts", "authors", "genres", "stypes", "words"}


def _int(v) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


@dataclass
class Selection:
    corpus_id: int | None = None
    texts: list[int] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    stypes: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    words: list[int] = field(default_factory=list)
    stop: str = ""
    stopset: frozenset = frozenset()  # стоп-слова и, при noprop, леммы имён собственных
    noprop: bool = False
    names: frozenset = frozenset()  # леммы имён собственных выборки (заполняются при noprop)
    corpus_ids: list[int] = field(default_factory=list)  # все готовые тексты корпуса
    ids: list[int] = field(default_factory=list)  # итоговая выборка после фильтров
    corpus_name: str = "Все тексты"

    @property
    def filtered(self) -> bool:
        return bool(self.texts or self.authors or self.genres or self.stypes or self.year_from or self.year_to)

    def pairs(self, with_words: bool = True) -> list[tuple[str, str]]:
        """Параметры выборки парами (ключ, значение) — для ссылок и скрытых полей форм."""
        out: list[tuple[str, str]] = []
        if self.corpus_id:
            out.append(("corpus_id", str(self.corpus_id)))
        for k in ("texts", "authors", "genres", "stypes"):
            out += [(k, str(v)) for v in getattr(self, k)]
        for k in ("year_from", "year_to"):
            if getattr(self, k):
                out.append((k, str(getattr(self, k))))
        if with_words:
            out += [("words", str(w)) for w in self.words]
        if self.stop:
            out.append(("stop", self.stop))
        if self.noprop:
            out.append(("noprop", "1"))
        return out

    def query(self, with_words: bool = True) -> str:
        return urlencode(self.pairs(with_words))

    def describe(self) -> str:
        parts = [f"корпус «{self.corpus_name}»"]
        if self.texts:
            parts.append(f"отмечено текстов: {len(self.texts)}")
        if self.authors:
            parts.append("авторы: " + ", ".join(self.authors))
        if self.genres:
            parts.append("жанры: " + ", ".join(self.genres))
        if self.stypes:
            from app.web.deps import SOURCE_TYPES

            parts.append("тип: " + ", ".join(SOURCE_TYPES.get(t, t) for t in self.stypes))
        if self.year_from or self.year_to:
            parts.append(f"годы: {self.year_from or '…'}–{self.year_to or '…'}")
        if self.noprop:
            parts.append(f"без имён собственных ({len(self.names)} лемм)")
        return "; ".join(parts)


def from_params(session: Session, qp, corpus_key: str = "corpus_id") -> Selection:
    """Собирает выборку из параметров запроса (Starlette QueryParams или dict списков)."""

    def getlist(k):
        if hasattr(qp, "getlist"):
            return qp.getlist(k)
        v = qp.get(k)
        return list(v) if isinstance(v, (list, tuple)) else ([v] if v else [])

    def get(k):
        if hasattr(qp, "getlist"):
            return qp.get(k)
        v = qp.get(k)
        return v[0] if isinstance(v, list) and v else (v or None)

    sel = Selection(
        corpus_id=_int(get(corpus_key)),
        texts=[int(x) for x in getlist("texts") if str(x).isdigit()],
        authors=[a for a in getlist("authors") if a],
        genres=[g for g in getlist("genres") if g],
        stypes=[t for t in getlist("stypes") if t],
        year_from=_int(get("year_from")),
        year_to=_int(get("year_to")),
        words=[int(x) for x in getlist("words") if str(x).isdigit()],
        stop=(get("stop") or "").strip(),
        noprop=bool(get("noprop")),
    )
    from app.services.stopwords import resolve

    sel.stopset = resolve(session, sel.stop)
    sel.corpus_ids = A.corpus_text_ids(session, sel.corpus_id)
    sel.corpus_name = A.corpus_label(session, sel.corpus_id)
    # отметки текстов другого корпуса (после смены корпуса) не должны давать пустую выборку
    sel.texts = [t for t in sel.texts if t in set(sel.corpus_ids)]
    base = [t for t in sel.corpus_ids if t in set(sel.texts)] if sel.texts else sel.corpus_ids
    sel.ids = A.filter_texts(session, base, year_from=sel.year_from, year_to=sel.year_to, authors=sel.authors,
                             genres=sel.genres, source_types=sel.stypes) if sel.filtered else base
    if sel.noprop:
        from app.services.entities import name_lemmas

        sel.names = name_lemmas(session, sel.ids)
        sel.stopset = sel.stopset | sel.names
    return sel


def corpus_texts(session: Session, sel: Selection) -> list[dict]:
    """Тексты корпуса для списка флажков в панели «Выборка»."""
    if not sel.corpus_ids:
        return []
    rows = session.execute(
        select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from, TextDoc.genre, TextDoc.source_type, TextDoc.n_words)
        .where(TextDoc.id.in_(sel.corpus_ids)).order_by(TextDoc.author, TextDoc.year_from, TextDoc.title)
    ).all()
    chosen = set(sel.ids)
    return [{"id": r.id, "title": r.title, "author": r.author or "", "year": r.year_from, "genre": r.genre or "",
             "stype": r.source_type, "n_words": r.n_words or 0, "checked": r.id in set(sel.texts),
             "in_selection": r.id in chosen} for r in rows]
