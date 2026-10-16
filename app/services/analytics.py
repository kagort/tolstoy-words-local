"""Запросы для анализа: частоты, конкорданс, профиль слова, сравнение, структура."""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import Integer, func, or_, select
from sqlalchemy.orm import Session

from app.models import (
    Corpus,
    DictionaryEntry,
    Hit,
    SentenceRow,
    TextDoc,
    TokenRelation,
    TokenRow,
    corpus_texts,
)
from app.nlp.syntax import ROLE_LABELS, SENT_TYPE_LABELS, rel_label_ud, role_label_ud
from app.nlp.tree import norm, parse_feats
from app.services.stats import SIGNIFICANCE_LL, gries_dp, ipm, log_dice, log_likelihood, log_ratio

VALID = or_(Hit.is_valid.is_(None), Hit.is_valid.is_(True))
DEFAULT_SCOPES = ("direct", "shared")


# ------------------------------------------------------------------ корпус


def corpus_text_ids(session: Session, corpus_id: int | None) -> list[int]:
    q = select(TextDoc.id).where(TextDoc.status == "ready")
    if corpus_id:
        q = q.join(corpus_texts, corpus_texts.c.text_id == TextDoc.id).where(corpus_texts.c.corpus_id == corpus_id)
    return list(session.scalars(q.order_by(TextDoc.id)))


def corpus_label(session: Session, corpus_id: int | None) -> str:
    if not corpus_id:
        return "Все тексты"
    c = session.get(Corpus, corpus_id)
    return c.name if c else f"Корпус {corpus_id}"


def corpus_words(session: Session, text_ids: list[int]) -> int:
    if not text_ids:
        return 0
    return int(session.scalar(select(func.coalesce(func.sum(TextDoc.n_words), 0)).where(TextDoc.id.in_(text_ids))) or 0)


def _hits(dictionary_id: int, text_ids: list[int], entry_id: int | None = None):
    q = select(Hit).where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(text_ids), VALID)
    if entry_id:
        q = q.where(Hit.entry_id == entry_id)
    return q


# ------------------------------------------------------------------ частоты


def frequency(session: Session, dictionary_id: int, text_ids: list[int]) -> dict:
    n_words = corpus_words(session, text_ids)
    sizes = dict(session.execute(select(TextDoc.id, TextDoc.n_words).where(TextDoc.id.in_(text_ids or [-1]))).all())
    entries = list(session.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dictionary_id).order_by(DictionaryEntry.id)))
    per_text = defaultdict(Counter)
    sent_counts: dict[int, int] = {}
    if text_ids:
        for entry_id, text_id, cnt in session.execute(
            select(Hit.entry_id, Hit.text_id, func.count())
            .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(text_ids), VALID)
            .group_by(Hit.entry_id, Hit.text_id)
        ):
            per_text[entry_id][text_id] = cnt
        sent_counts = dict(session.execute(
            select(Hit.entry_id, func.count(func.distinct(Hit.sentence_id)))
            .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(text_ids), VALID)
            .group_by(Hit.entry_id)
        ).all())
    rows = []
    for e in entries:
        counts = per_text.get(e.id, Counter())
        total = sum(counts.values())
        rows.append({
            "entry_id": e.id,
            "word": e.surface,
            "lemma": e.lemma,
            "upos": e.upos or "",
            "category": e.category or "",
            "count": total,
            "ipm": round(ipm(total, n_words), 2),
            "sentences": sent_counts.get(e.id, 0),
            "texts": len(counts),
            "dp": round(gries_dp([counts.get(t, 0) for t in text_ids], [sizes.get(t, 0) for t in text_ids]), 3) if total else None,
        })
    rows.sort(key=lambda r: -r["count"])
    return {"rows": rows, "corpus_words": n_words, "n_texts": len(text_ids)}


def frequency_by_text(session: Session, dictionary_id: int, text_ids: list[int], top: int = 15) -> dict:
    """Матрица ipm «слово × текст» для тепловой карты."""
    texts = list(session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.n_words).where(TextDoc.id.in_(text_ids or [-1])).order_by(TextDoc.id)))
    freq = frequency(session, dictionary_id, text_ids)["rows"][:top]
    ids = [r["entry_id"] for r in freq]
    counts = defaultdict(dict)
    if ids:
        for entry_id, text_id, cnt in session.execute(
            select(Hit.entry_id, Hit.text_id, func.count())
            .where(Hit.entry_id.in_(ids), Hit.text_id.in_(text_ids), VALID)
            .group_by(Hit.entry_id, Hit.text_id)
        ):
            counts[entry_id][text_id] = cnt
    return {
        "words": [r["word"] for r in freq],
        "texts": [f"{t.title}" + (f" ({t.author})" if t.author else "") for t in texts],
        "z": [[round(ipm(counts[e].get(t.id, 0), t.n_words), 1) for t in texts] for e in ids],
    }


def forms_distribution(session: Session, entry_id: int, text_ids: list[int]) -> dict:
    rows = session.execute(
        select(TokenRow.form_norm, TokenRow.feats)
        .join(Hit, Hit.token_id == TokenRow.id)
        .where(Hit.entry_id == entry_id, Hit.text_id.in_(text_ids or [-1]), VALID)
    ).all()
    forms = Counter(f for f, _ in rows)
    feats = defaultdict(Counter)
    for _, fs in rows:
        for k, v in parse_feats(fs).items():
            if k in ("Case", "Number", "Tense", "Aspect", "Gender", "VerbForm", "Mood", "Voice", "Degree", "Variant"):
                feats[k][v] += 1
    return {"forms": forms.most_common(30), "feats": {k: c.most_common() for k, c in sorted(feats.items())}}


# ------------------------------------------------------------------ конкорданс


def concordance(
    session: Session,
    dictionary_id: int,
    text_ids: list[int],
    *,
    entry_id: int | None = None,
    entry_ids: list[int] | None = None,
    role: str | None = None,
    rel_type: str | None = None,
    related_lemma: str | None = None,
    include_invalid: bool = False,
    page: int = 1,
    per_page: int = 50,
    context_chars: int = 90,
) -> dict:
    q = (
        select(Hit.id, Hit.sentence_id, Hit.role, Hit.role_subtype, Hit.is_valid, Hit.form,
               TokenRow.start_char, TokenRow.end_char, TokenRow.deprel, TokenRow.upos, SentenceRow.text, SentenceRow.idx,
               TextDoc.id.label("text_id"), TextDoc.title, TextDoc.author, TextDoc.year_from,
               DictionaryEntry.surface)
        .join(TokenRow, TokenRow.id == Hit.token_id)
        .join(SentenceRow, SentenceRow.id == Hit.sentence_id)
        .join(TextDoc, TextDoc.id == Hit.text_id)
        .join(DictionaryEntry, DictionaryEntry.id == Hit.entry_id)
        .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(text_ids or [-1]))
    )
    if not include_invalid:
        q = q.where(VALID)
    if entry_id:
        q = q.where(Hit.entry_id == entry_id)
    elif entry_ids:
        q = q.where(Hit.entry_id.in_(entry_ids))
    if role:
        q = q.where(Hit.role == role)
    if rel_type or related_lemma:
        sub = select(TokenRelation.hit_id).where(TokenRelation.dictionary_id == dictionary_id)
        if rel_type:
            sub = sub.where(TokenRelation.rel_type == rel_type)
        if related_lemma:
            sub = sub.where(TokenRelation.related_lemma == related_lemma.lower())
        q = q.where(Hit.id.in_(sub))
    total = session.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = session.execute(q.order_by(TextDoc.id, SentenceRow.idx, Hit.token_idx).offset((page - 1) * per_page).limit(per_page)).all()
    items = []
    for r in rows:
        text = r.text
        left = text[: r.start_char]
        right = text[r.end_char :]
        items.append({
            "hit_id": r.id,
            "sentence_id": r.sentence_id,
            "left": ("…" + left[-context_chars:]) if len(left) > context_chars else left,
            "kw": text[r.start_char : r.end_char],
            "right": (right[:context_chars] + "…") if len(right) > context_chars else right,
            "sentence": text,
            "role": r.role,
            "role_label": ROLE_LABELS.get(r.role, r.role),
            "deprel": r.deprel,
            "upos": r.upos,
            "start": r.start_char,
            "end": r.end_char,
            "role_subtype": r.role_subtype or "",
            "is_valid": r.is_valid,
            "word": r.surface,
            "text_id": r.text_id,
            "source": f"{r.author + ', ' if r.author else ''}«{r.title}»{', ' + str(r.year_from) if r.year_from else ''}",
            "sentence_no": r.idx,
        })
    return {"items": items, "total": total, "page": page, "per_page": per_page, "pages": max(1, -(-total // per_page))}


# ------------------------------------------------------------------ профиль слова


def word_profile(session: Session, entry_id: int, text_ids: list[int], scopes=DEFAULT_SCOPES,
                 top: int = 25, include_negated: bool = False, stop: frozenset[str] = frozenset()) -> dict:
    """stop — стоп-слова: такие связанные леммы не показываются (данные в базе не меняются)."""
    entry = session.get(DictionaryEntry, entry_id)
    if entry is None or not text_ids:
        return {"entry": entry, "total": 0, "roles": [], "relations": {}}
    hit_q = _hits(entry.dictionary_id, text_ids, entry_id)
    total = session.scalar(select(func.count()).select_from(hit_q.subquery())) or 0
    roles = session.execute(
        select(Hit.role, func.count()).where(Hit.entry_id == entry_id, Hit.text_id.in_(text_ids), VALID).group_by(Hit.role).order_by(func.count().desc())
    ).all()
    rel_q = (
        select(TokenRelation.rel_type, TokenRelation.related_lemma, func.count().label("n"),
               func.count(func.distinct(TokenRelation.text_id)).label("texts"),
               func.min(Hit.sentence_id).label("example"))
        .join(Hit, Hit.id == TokenRelation.hit_id)
        .where(TokenRelation.entry_id == entry_id, TokenRelation.text_id.in_(text_ids), VALID,
               TokenRelation.scope.in_(list(scopes)))
        .group_by(TokenRelation.rel_type, TokenRelation.related_lemma)
    )
    if not include_negated:
        rel_q = rel_q.where(TokenRelation.negated.is_(False))
    grouped = defaultdict(list)
    for r in session.execute(rel_q):
        if stop and r.rel_type != "case" and norm(r.related_lemma) in stop:
            continue
        grouped[r.rel_type].append({"lemma": r.related_lemma, "count": r.n, "texts": r.texts, "example": r.example})
    # частоты связанных лемм в корпусе для logDice
    lemmas = {x["lemma"] for rt, xs in grouped.items() if rt != "case" for x in xs}
    fy = {}
    if lemmas:
        fy = dict(session.execute(
            select(TokenRow.lemma_norm, func.count())
            .where(TokenRow.lemma_norm.in_([norm(x) for x in lemmas]), TokenRow.text_id.in_(text_ids))
            .group_by(TokenRow.lemma_norm)
        ).all())
    relations = {}
    order = ["attribute", "predicate", "governor", "subject_of_pred", "adverbial", "conj", "case"]
    for rt in sorted(grouped, key=lambda k: order.index(k) if k in order else 99):
        items = grouped[rt]
        for x in items:
            x["logdice"] = round(log_dice(x["count"], total, fy.get(norm(x["lemma"]), x["count"])), 2) if rt != "case" else None
        items.sort(key=lambda x: (-x["count"], -(x["logdice"] or 0)))
        relations[rt] = {"label": rel_label_ud(rt), "items": items[:top], "distinct": len(items),
                         "total": sum(x["count"] for x in items)}
    return {
        "entry": entry,
        "total": total,
        "roles": [{"role": r, "label": role_label_ud(r), "count": c} for r, c in roles],
        "relations": relations,
    }


# ------------------------------------------------------------------ сравнение корпусов


def compare_frequency(session: Session, dictionary_id: int, ids_a: list[int], ids_b: list[int]) -> dict:
    fa = {r["entry_id"]: r for r in frequency(session, dictionary_id, ids_a)["rows"]}
    fb = {r["entry_id"]: r for r in frequency(session, dictionary_id, ids_b)["rows"]}
    na, nb = corpus_words(session, ids_a), corpus_words(session, ids_b)
    rows = []
    for eid, ra in fa.items():
        rb = fb.get(eid, {"count": 0, "ipm": 0})
        a, b = ra["count"], rb["count"]
        ll = log_likelihood(a, b, na, nb)
        rows.append({
            "entry_id": eid, "word": ra["word"], "count_a": a, "count_b": b, "ipm_a": ra["ipm"], "ipm_b": rb["ipm"],
            "ll": round(ll, 2), "log_ratio": round(log_ratio(a, b, na, nb), 2), "significant": ll >= SIGNIFICANCE_LL,
        })
    rows.sort(key=lambda r: -r["ll"])
    return {"rows": rows, "words_a": na, "words_b": nb}


def compare_profile(session: Session, entry_id: int, ids_a: list[int], ids_b: list[int], rel_type: str = "attribute",
                    scopes=DEFAULT_SCOPES, top: int = 40, stop: frozenset[str] = frozenset()) -> dict:
    def counts(ids):
        if not ids:
            return Counter(), 0
        rows = session.execute(
            select(TokenRelation.related_lemma, func.count())
            .join(Hit, Hit.id == TokenRelation.hit_id)
            .where(TokenRelation.entry_id == entry_id, TokenRelation.text_id.in_(ids), TokenRelation.rel_type == rel_type,
                   TokenRelation.scope.in_(list(scopes)), TokenRelation.negated.is_(False), VALID)
            .group_by(TokenRelation.related_lemma)
        ).all()
        hits = session.scalar(select(func.count()).select_from(Hit).where(Hit.entry_id == entry_id, Hit.text_id.in_(ids), VALID)) or 0
        return Counter(dict(rows)), hits

    ca, ha = counts(ids_a)
    cb, hb = counts(ids_b)
    rows = []
    for lemma in set(ca) | set(cb):
        if stop and rel_type != "case" and norm(lemma) in stop:
            continue
        a, b = ca.get(lemma, 0), cb.get(lemma, 0)
        ll = log_likelihood(a, b, max(ha, 1), max(hb, 1))
        rows.append({"lemma": lemma, "count_a": a, "count_b": b,
                     "per100_a": round(100 * a / ha, 1) if ha else 0, "per100_b": round(100 * b / hb, 1) if hb else 0,
                     "ll": round(ll, 2), "log_ratio": round(log_ratio(a, b, max(ha, 1), max(hb, 1)), 2),
                     "significant": ll >= SIGNIFICANCE_LL})
    rows.sort(key=lambda r: (-r["ll"], -(r["count_a"] + r["count_b"])))
    return {"rows": rows[:top], "hits_a": ha, "hits_b": hb, "rel_label": rel_label_ud(rel_type)}


# ------------------------------------------------------------------ структура предложений


def _structure_stats(session: Session, where) -> dict:
    """Агрегаты считаются в SQL, чтобы не тянуть миллион предложений в Python."""
    S = SentenceRow
    agg = session.execute(
        select(
            func.count(),
            func.avg(S.n_words),
            func.percentile_cont(0.5).within_group(S.n_words),
            func.stddev_pop(S.n_words),
            func.avg(S.tree_depth),
            func.avg(S.n_clauses),
            func.avg(func.cast(S.is_direct_speech, Integer)),
            func.avg(func.cast(S.has_homogeneous, Integer)),
            func.avg(func.cast(S.has_participial, Integer)),
            func.avg(func.cast(S.has_gerund, Integer)),
        ).where(where)
    ).one()
    n = agg[0] or 0
    if not n:
        return {"n": 0}
    types = session.execute(select(S.sent_type, func.count()).where(where).group_by(S.sent_type).order_by(func.count().desc())).all()
    bucket = func.least(S.n_words / 5 * 5, 100)
    hist = session.execute(select(bucket, func.count()).where(where).group_by(bucket).order_by(bucket)).all()

    def pct(x):
        return round(100 * float(x or 0), 1)

    return {
        "n": n,
        "avg_len": round(float(agg[1]), 2),
        "median_len": float(agg[2]),
        "stdev_len": round(float(agg[3] or 0), 2),
        "avg_depth": round(float(agg[4]), 2),
        "avg_clauses": round(float(agg[5]), 2),
        "types": {SENT_TYPE_LABELS.get(k, k): round(100 * v / n, 1) for k, v in types},
        "direct_speech": pct(agg[6]),
        "homogeneous": pct(agg[7]),
        "participial": pct(agg[8]),
        "gerund": pct(agg[9]),
        # распределение длины (в % предложений), корзины по 5 слов; 100 = «100 и более»
        "hist": {int(b): round(100 * c / n, 2) for b, c in hist},
    }


def sentence_structure(session: Session, dictionary_id: int, text_ids: list[int], entry_id: int | None = None,
                       entry_ids: list[int] | None = None) -> dict:
    ids = text_ids or [-1]
    hit_sids = select(Hit.sentence_id).where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(ids), VALID)
    if entry_id:
        hit_sids = hit_sids.where(Hit.entry_id == entry_id)
    elif entry_ids:
        hit_sids = hit_sids.where(Hit.entry_id.in_(entry_ids))
    return {
        "hits": _structure_stats(session, SentenceRow.id.in_(hit_sids)),
        "all": _structure_stats(session, SentenceRow.text_id.in_(ids)),
    }


# ------------------------------------------------------------------ обзор по текстам / авторам / жанрам / периодам

DIMENSIONS = {"author": "Автор", "text": "Текст", "genre": "Жанр", "source_type": "Тип текста", "period": "Период"}
PERIOD_STEPS = (1, 5, 10, 25, 50, 100)
NO_DATE = -1


def facets(session: Session, text_ids: list[int]) -> dict:
    """Значения для фильтров обзора: авторы, жанры, диапазон лет."""
    ids = text_ids or [-1]
    authors = [a for a in session.scalars(select(TextDoc.author).where(TextDoc.id.in_(ids)).distinct().order_by(TextDoc.author)) if a]
    genres = [g for g in session.scalars(select(TextDoc.genre).where(TextDoc.id.in_(ids)).distinct().order_by(TextDoc.genre)) if g]
    y_min, y_max = session.execute(select(func.min(TextDoc.year_from), func.max(TextDoc.year_from)).where(TextDoc.id.in_(ids))).one()
    return {"authors": authors, "genres": genres, "year_min": y_min, "year_max": y_max}


def filter_texts(session: Session, text_ids: list[int], *, year_from: int | None = None, year_to: int | None = None,
                 authors: list[str] | None = None, genres: list[str] | None = None,
                 source_types: list[str] | None = None) -> list[int]:
    """Сужает выборку текстов по годам, авторам, жанрам, типу. Тексты без года при фильтре по годам исключаются."""
    q = select(TextDoc.id).where(TextDoc.id.in_(text_ids or [-1]))
    if year_from:
        q = q.where(TextDoc.year_from >= year_from)
    if year_to:
        q = q.where(TextDoc.year_from <= year_to)
    if authors:
        q = q.where(TextDoc.author.in_(authors))
    if genres:
        q = q.where(TextDoc.genre.in_(genres))
    if source_types:
        q = q.where(TextDoc.source_type.in_(source_types))
    return list(session.scalars(q.order_by(TextDoc.id)))


def _group_expr(dim: str, step: int):
    if dim == "text":
        return TextDoc.id
    if dim == "author":
        return func.coalesce(TextDoc.author, "—")
    if dim == "genre":
        return func.coalesce(TextDoc.genre, "—")
    if dim == "source_type":
        return TextDoc.source_type
    # period: начало интервала; тексты без года → NO_DATE
    return func.coalesce(func.floor(TextDoc.year_from / float(step)) * step, NO_DATE)


def _period_label(start: int, step: int) -> str:
    if start == NO_DATE:
        return "без даты"
    return str(start) if step == 1 else f"{start}–{start + step - 1}"


def overview(session: Session, dictionary_id: int, text_ids: list[int], entry_ids: list[int],
             dim: str = "author", step: int = 10) -> dict:
    """Матрица «группа × слово»: вхождения, ipm, доля текстов группы со словом; плюс сумма по выбранным словам."""
    from app.web.deps import SOURCE_TYPES  # подписи типов текста

    dim = dim if dim in DIMENSIONS else "author"
    step = step if step in PERIOD_STEPS else 10
    ids = text_ids or [-1]
    key = _group_expr(dim, step).label("g")
    sizes = session.execute(
        select(key, func.coalesce(func.sum(TextDoc.n_words), 0), func.count(TextDoc.id)).where(TextDoc.id.in_(ids)).group_by(key)
    ).all()
    groups = []
    titles = {}
    if dim == "text":
        titles = {t.id: (f"{t.author}, " if t.author else "") + f"«{t.title}»" + (f" ({t.year_from})" if t.year_from else "")
                  for t in session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from).where(TextDoc.id.in_(ids)))}
    for g, n_words, n_texts in sizes:
        g = int(g) if dim in ("text", "period") and g is not None else g
        if dim == "text":
            label = titles.get(g, str(g))
        elif dim == "period":
            label = _period_label(g, step)
        elif dim == "source_type":
            label = SOURCE_TYPES.get(g, g)
        else:
            label = g
        groups.append({"key": g, "label": label, "n_words": int(n_words or 0), "n_texts": n_texts})
    if dim == "period":
        groups.sort(key=lambda x: (x["key"] == NO_DATE, x["key"]))
    else:
        groups.sort(key=lambda x: -x["n_words"])

    entries = list(session.scalars(select(DictionaryEntry).where(DictionaryEntry.id.in_(entry_ids or [-1]))))
    order = {eid: k for k, eid in enumerate(entry_ids)}
    entries.sort(key=lambda e: order.get(e.id, 0))
    counts, texts_with = defaultdict(dict), defaultdict(dict)
    total_counts, total_texts = {}, {}
    if entries:
        base = (select(key, Hit.entry_id, func.count(), func.count(func.distinct(Hit.text_id)))
                .join(TextDoc, TextDoc.id == Hit.text_id)
                .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(ids), Hit.entry_id.in_([e.id for e in entries]), VALID))
        for g, eid, c, t in session.execute(base.group_by(key, Hit.entry_id)):
            g = int(g) if dim in ("text", "period") and g is not None else g
            counts[eid][g], texts_with[eid][g] = c, t
        tot = (select(key, func.count(), func.count(func.distinct(Hit.text_id)))
               .join(TextDoc, TextDoc.id == Hit.text_id)
               .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(ids), Hit.entry_id.in_([e.id for e in entries]), VALID)
               .group_by(key))
        for g, c, t in session.execute(tot):
            g = int(g) if dim in ("text", "period") and g is not None else g
            total_counts[g], total_texts[g] = c, t

    def row(cmap, tmap):
        return {
            "count": [cmap.get(gr["key"], 0) for gr in groups],
            "ipm": [round(ipm(cmap.get(gr["key"], 0), gr["n_words"]), 2) for gr in groups],
            "share": [round(100 * tmap.get(gr["key"], 0) / gr["n_texts"], 1) if gr["n_texts"] else 0 for gr in groups],
        }

    words = [{"entry_id": e.id, "word": e.surface, **row(counts[e.id], texts_with[e.id])} for e in entries]
    total = {"word": "все выбранные слова", **row(total_counts, total_texts)}
    return {"dim": dim, "dim_label": DIMENSIONS[dim], "step": step, "groups": groups, "words": words, "total": total,
            "corpus_words": sum(g["n_words"] for g in groups), "n_texts": sum(g["n_texts"] for g in groups)}
