"""Анализ пунктуации выборки: частоты знаков на 1000 слов по группам (тексты, авторы, жанры, десятилетия), профиль
пунктуации, как заканчиваются предложения, запятые на предложение, доля прямой речи, примеры предложений со знаком.

Знак — токен с частью речи PUNCT (разбор не нужен точный: знаки одинаковы в предварительном и точном разборе).
Сочетания токенизатора («?!», «!..», «?..», «...») раскладываются по смыслу: см. classify().
"""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.models import SentenceRow, TextDoc, TokenRow
from app.services.labels import short_author, short_title

MARKS = {  # ключ → (подпись, короткая подпись для графика)
    "comma": ("запятая ,", ","), "period": ("точка .", "."), "dash": ("тире —", "—"), "colon": ("двоеточие :", ":"),
    "semicolon": ("точка с запятой ;", ";"), "ellipsis": ("многоточие …", "…"), "exclam": ("восклицательный !", "!"),
    "question": ("вопросительный ?", "?"), "qe": ("вопрос + восклицание ?!", "?!"), "quote": ("кавычки «» (пар)", "«»"),
    "paren": ("скобки () (пар)", "()"),
}
ENDINGS = {"period": "повествовательные .", "question": "вопросительные ?", "exclam": "восклицательные !",
           "ellipsis": "многоточие …", "qe": "?! / !?", "none": "без знака в конце"}
CLOSING = ("»", "“", "”", '"', "'", "’", ")", "]")  # не конец предложения — смотрим знак перед ними
DIMS = {"text": "Текст", "author": "Автор", "genre": "Жанр", "period": "Десятилетие"}
TOP_GROUPS = 30


def classify(form: str) -> str | None:
    f = (form or "").strip()
    if not f:
        return None
    if f == ",":
        return "comma"
    if ("?" in f and "!" in f):
        return "qe"
    if f.startswith("?"):
        return "question"
    if f.startswith("!"):
        return "exclam"
    if f.startswith("…") or (len(f) > 1 and set(f) == {"."}):  # «…», «...», «…..»
        return "ellipsis"
    if f == ".":
        return "period"
    if f in ("—", "–", "-", "--", "―"):
        return "dash"
    if f == ":":
        return "colon"
    if f == ";":
        return "semicolon"
    if f in ("«", "»", "„", "“", "”", '"'):
        return "quote"
    if f in ("(", ")"):
        return "paren"
    return None


def _group_of(doc, dim: str) -> tuple[str, str]:
    if dim == "author":
        a = short_author(doc.author) if doc.author else "без автора"
        return a, doc.author or "без автора"
    if dim == "genre":
        g = doc.genre or "без жанра"
        return g, g
    if dim == "period":
        if not doc.year_from:
            return "без даты", "без даты"
        d = doc.year_from // 10 * 10
        return f"{d}-е", f"{d}–{d + 9}"
    return short_title(doc.title) if doc.title else f"текст {doc.id}", f"{short_author(doc.author) + ', ' if doc.author else ''}«{doc.title}»"


def analyze(session: Session, text_ids: list[int], dim: str = "author") -> dict:
    dim = dim if dim in DIMS else "author"
    docs = {d.id: d for d in session.scalars(select(TextDoc).where(TextDoc.id.in_(text_ids or [-1])))}
    if not docs:
        return {"dim": dim, "dim_label": DIMS[dim], "groups": [], "marks": MARKS, "endings": ENDINGS}
    ids = list(docs)
    # знаки по текстам
    marks: dict[int, Counter] = defaultdict(Counter)
    for tid, form, n in session.execute(select(TokenRow.text_id, TokenRow.form, func.count())
                                        .where(TokenRow.text_id.in_(ids), TokenRow.upos == "PUNCT")
                                        .group_by(TokenRow.text_id, TokenRow.form)):
        k = classify(form)
        if k:
            marks[tid][k] += n
    # концовки: последний токен предложения, не считая закрывающих кавычек и скобок
    last = (select(TokenRow.sentence_id.label("sid"), func.max(TokenRow.idx).label("m"))
            .where(TokenRow.text_id.in_(ids), TokenRow.form.not_in(CLOSING)).group_by(TokenRow.sentence_id).subquery())
    endings: dict[int, Counter] = defaultdict(Counter)
    for tid, form, upos in session.execute(select(TokenRow.text_id, TokenRow.form, TokenRow.upos)
                                           .join(last, and_(TokenRow.sentence_id == last.c.sid, TokenRow.idx == last.c.m))):
        k = classify(form) if upos == "PUNCT" else None
        endings[tid][k if k in ENDINGS else "none"] += 1
    # предложения, прямая речь
    sent: dict[int, tuple[int, int]] = {}
    for tid, n, ds in session.execute(select(SentenceRow.text_id, func.count(), func.count().filter(SentenceRow.is_direct_speech))
                                      .where(SentenceRow.text_id.in_(ids)).group_by(SentenceRow.text_id)):
        sent[tid] = (n, ds)
    # по группам
    agg: dict[str, dict] = {}
    for tid, doc in docs.items():
        key, full = _group_of(doc, dim)
        g = agg.setdefault(key, {"label": key, "full": full, "words": 0, "sentences": 0, "direct": 0, "texts": 0,
                                 "marks": Counter(), "endings": Counter()})
        g["words"] += doc.n_words or 0
        g["texts"] += 1
        n, ds = sent.get(tid, (0, 0))
        g["sentences"] += n
        g["direct"] += ds
        g["marks"].update(marks.get(tid, {}))
        g["endings"].update(endings.get(tid, {}))
    groups = sorted(agg.values(), key=lambda g: (g["label"] if dim == "period" else "", -g["words"]))
    hidden = 0
    if dim != "period" and len(groups) > TOP_GROUPS:
        hidden = len(groups) - TOP_GROUPS
        groups = groups[:TOP_GROUPS]
    total = {"label": "вся выборка", "full": "вся выборка", "words": sum(d.n_words or 0 for d in docs.values()),
             "sentences": sum(v[0] for v in sent.values()), "direct": sum(v[1] for v in sent.values()), "texts": len(docs),
             "marks": sum((marks[t] for t in ids), Counter()), "endings": sum((endings[t] for t in ids), Counter())}
    out_groups = [_row(g) for g in groups]
    return {"dim": dim, "dim_label": DIMS[dim], "groups": out_groups, "total": _row(total), "hidden": hidden,
            "marks": MARKS, "endings": ENDINGS, "mark_order": list(MARKS), "ending_order": list(ENDINGS)}


def _row(g: dict) -> dict:
    w, s = g["words"] or 0, g["sentences"] or 0
    m = g["marks"]
    per1000 = {k: round(1000 * (m.get(k, 0) / (2 if k in ("quote", "paren") else 1)) / w, 2) if w else 0 for k in MARKS}
    ends = g["endings"]
    n_end = sum(ends.values()) or 1
    return {"label": g["label"], "full": g["full"], "words": w, "sentences": s, "texts": g["texts"],
            "per1000": per1000, "counts": {k: m.get(k, 0) for k in MARKS},
            "endings": {k: round(100 * ends.get(k, 0) / n_end, 1) for k in ENDINGS},
            "commas_per_sentence": round(m.get("comma", 0) / s, 2) if s else 0,
            "words_per_sentence": round(w / s, 1) if s else 0,
            "direct_pct": round(100 * g["direct"] / s, 1) if s else 0}


def examples(session: Session, text_ids: list[int], mark: str, page: int = 1, per_page: int = 30) -> dict:
    """Предложения выборки с этим знаком (по порядку текстов) — с подсветкой знака."""
    if mark not in MARKS or not text_ids:
        return {"items": [], "total": 0, "pages": 1, "page": 1}
    forms = [f for f, in session.execute(select(TokenRow.form).where(TokenRow.text_id.in_(text_ids), TokenRow.upos == "PUNCT")
                                         .group_by(TokenRow.form)) if classify(f) == mark]
    if not forms:
        return {"items": [], "total": 0, "pages": 1, "page": 1}
    has = (select(TokenRow.sentence_id).where(TokenRow.text_id.in_(text_ids), TokenRow.upos == "PUNCT", TokenRow.form.in_(forms))
           .distinct().subquery())
    total = session.scalar(select(func.count()).select_from(has)) or 0
    rows = session.execute(select(SentenceRow.id, SentenceRow.text, SentenceRow.idx, TextDoc.title, TextDoc.author)
                           .join(has, has.c.sentence_id == SentenceRow.id).join(TextDoc, TextDoc.id == SentenceRow.text_id)
                           .order_by(SentenceRow.text_id, SentenceRow.idx).offset((page - 1) * per_page).limit(per_page)).all()
    spans: dict[int, list] = defaultdict(list)
    for sid, st, en in session.execute(select(TokenRow.sentence_id, TokenRow.start_char, TokenRow.end_char)
                                       .where(TokenRow.sentence_id.in_([r.id for r in rows] or [-1]), TokenRow.upos == "PUNCT",
                                              TokenRow.form.in_(forms))):
        spans[sid].append((st, en, "kw"))
    from app.services.portrait import marked

    items = [{"sentence_id": r.id, "parts": marked(r.text, spans.get(r.id, [])), "sent_idx": r.idx,
              "source": f"{short_author(r.author) + ', ' if r.author else ''}«{r.title}»"} for r in rows]
    return {"items": items, "total": total, "page": page, "pages": max(1, -(-total // per_page))}
