"""Ключевые слова выборки относительно опорного корпуса (keyness) и TF-IDF для каждого текста."""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import SentenceRow, TextDoc, TokenRow
from app.services import analytics as A
from app.services.collocations import POS_GROUPS
from app.services.stats import SIGNIFICANCE_LL, ipm, log_likelihood, log_ratio, tf_idf

MAX_TFIDF_TEXTS = 40


def _lemma_counts(session: Session, text_ids: list[int], pos_set) -> tuple[Counter, dict[str, str]]:
    q = select(TokenRow.lemma_norm, TokenRow.upos, func.count()).where(TokenRow.text_id.in_(text_ids or [-1]))
    q = q.where(TokenRow.upos.in_(pos_set)) if pos_set else q.where(TokenRow.upos != "PUNCT")
    counts: Counter = Counter()
    by_pos: dict[str, Counter] = defaultdict(Counter)
    for lemma, upos, c in session.execute(q.group_by(TokenRow.lemma_norm, TokenRow.upos)):
        if not lemma or not any(ch.isalpha() for ch in lemma):
            continue
        counts[lemma] += c
        by_pos[lemma][upos] += c
    return counts, {lemma: c.most_common(1)[0][0] for lemma, c in by_pos.items()}


def reference_ids(session: Session, text_ids: list[int], ref: str | None) -> tuple[list[int], str]:
    """Опорный корпус: "" — все остальные тексты базы; число — корпус (без текстов выборки)."""
    target = set(text_ids)
    if ref and ref.isdigit():
        ids = [t for t in A.corpus_text_ids(session, int(ref)) if t not in target]
        return ids, f"корпус «{A.corpus_label(session, int(ref))}» (без текстов выборки)"
    return [t for t in A.corpus_text_ids(session, None) if t not in target], "все остальные тексты базы"


def keywords(session: Session, text_ids: list[int], ref_ids: list[int], *, pos: str = "content", min_count: int = 3,
             stop: frozenset[str] = frozenset(), top: int = 100) -> dict:
    """Положительные (чаще в выборке) и отрицательные (реже) ключевые слова по G² и log-ratio."""
    pos_set = POS_GROUPS.get(pos, POS_GROUPS["content"])[1]
    ta, pos_a = _lemma_counts(session, text_ids, pos_set)
    tb, pos_b = _lemma_counts(session, ref_ids, pos_set) if ref_ids else (Counter(), {})
    na, nb = A.corpus_words(session, text_ids), A.corpus_words(session, ref_ids)
    pos_main = {**pos_b, **pos_a}
    positive, negative = [], []
    for lemma in set(ta) | set(tb):
        if lemma in stop:
            continue
        a, b = ta.get(lemma, 0), tb.get(lemma, 0)
        if a + b < min_count:
            continue
        ll = log_likelihood(a, b, na, nb) if nb else 0.0
        row = {"lemma": lemma, "pos": pos_main.get(lemma, ""), "count_a": a, "count_b": b,
               "ipm_a": round(ipm(a, na), 1), "ipm_b": round(ipm(b, nb), 1), "ll": round(ll, 2),
               "log_ratio": round(log_ratio(a, b, na, nb), 2) if nb else None, "significant": ll >= SIGNIFICANCE_LL}
        if not nb:
            positive.append(row)
        elif row["ipm_a"] > row["ipm_b"]:
            positive.append(row)
        else:
            negative.append(row)
    key = (lambda r: -r["count_a"]) if not nb else (lambda r: -r["ll"])
    positive.sort(key=key)
    negative.sort(key=key)
    return {"positive": positive[:top], "negative": negative[:top], "words_a": na, "words_b": nb,
            "n_positive": sum(1 for r in positive if r["significant"]), "n_negative": sum(1 for r in negative if r["significant"])}


CONTEXT_UNITS = {"sent": "предложение", "win": "предложение ±1", "para": "абзац"}
MAX_CONTEXT_SENTENCES = 50_000


def context_sentences(session: Session, sentence_ids: list[int], unit: str = "sent") -> list[int]:
    """Предложения контекстов: сами найденные, с соседними (±1) или целыми абзацами."""
    if unit not in ("win", "para") or not sentence_ids:
        return sorted(set(sentence_ids))
    out: set[int] = set()
    for k in range(0, len(sentence_ids), 5000):
        found = select(SentenceRow.text_id, SentenceRow.idx, SentenceRow.paragraph_idx).where(
            SentenceRow.id.in_(sentence_ids[k: k + 5000])).subquery()
        cond = (SentenceRow.paragraph_idx == found.c.paragraph_idx) if unit == "para" else SentenceRow.idx.between(found.c.idx - 1, found.c.idx + 1)
        out.update(session.scalars(select(SentenceRow.id).join(found, (SentenceRow.text_id == found.c.text_id) & cond)))
    return sorted(out)


def context_keywords(session: Session, sentence_ids: list[int], text_ids: list[int], *, unit: str = "sent",
                     node_tokens: list[int] | None = None, exclude: set[str] | frozenset[str] = frozenset(),
                     pos: str = "content", min_count: int = 3, stop: frozenset[str] = frozenset(), top: int = 100) -> dict:
    """Ключевые слова контекстов (мини-корпус найденных предложений/абзацев) против остальной выборки.

    Леммы самих вхождений (node_tokens) и запроса (exclude) не показываются: иначе они всегда наверху.
    """
    ctx_ids = context_sentences(session, sentence_ids, unit)
    too_many = len(ctx_ids) > MAX_CONTEXT_SENTENCES
    pos_set = POS_GROUPS.get(pos, POS_GROUPS["content"])[1]
    node = set(node_tokens or ())
    ta: Counter = Counter()
    by_pos: dict[str, Counter] = defaultdict(Counter)
    na = 0
    if ctx_ids and not too_many:
        for k in range(0, len(ctx_ids), 5000):
            chunk = ctx_ids[k: k + 5000]
            na += int(session.scalar(select(func.coalesce(func.sum(SentenceRow.n_words), 0)).where(SentenceRow.id.in_(chunk))) or 0)
            q = select(TokenRow.id, TokenRow.lemma_norm, TokenRow.upos).where(TokenRow.sentence_id.in_(chunk))
            q = q.where(TokenRow.upos.in_(pos_set)) if pos_set else q.where(TokenRow.upos != "PUNCT")
            for tid, lemma, upos in session.execute(q):
                if tid in node:
                    exclude = exclude | {lemma}
                    continue
                if not lemma or not any(ch.isalpha() for ch in lemma):
                    continue
                ta[lemma] += 1
                by_pos[lemma][upos] += 1
    total, pos_all = _lemma_counts(session, text_ids, pos_set) if ta else (Counter(), {})
    nb = max(0, A.corpus_words(session, text_ids) - na) if ta else 0
    rows = []
    for lemma, a in ta.items():
        if lemma in stop or lemma in exclude or a < min_count:
            continue
        b = max(0, total.get(lemma, 0) - a)
        ll = log_likelihood(a, b, na, nb) if nb else 0.0
        ia, ib = ipm(a, na), ipm(b, nb)
        if nb and ia <= ib:
            continue
        rows.append({"lemma": lemma, "pos": by_pos[lemma].most_common(1)[0][0] if by_pos[lemma] else pos_all.get(lemma, ""),
                     "count_a": a, "count_b": b, "ipm_a": round(ia, 1), "ipm_b": round(ib, 1), "ll": round(ll, 2),
                     "log_ratio": round(log_ratio(a, b, na, nb), 2) if nb else None, "significant": ll >= SIGNIFICANCE_LL})
    rows.sort(key=(lambda r: -r["ll"]) if nb else (lambda r: -r["count_a"]))
    return {"positive": rows[:top], "n_positive": sum(1 for r in rows if r["significant"]), "words_a": na, "words_b": nb,
            "n_sentences": len(sentence_ids), "n_context": len(ctx_ids), "unit": unit, "too_many": too_many}


def tfidf_by_text(session: Session, text_ids: list[int], *, pos: str = "content", stop: frozenset[str] = frozenset(),
                  top: int = 12) -> dict:
    """Самые характерные леммы каждого текста выборки: TF-IDF, где коллекция — все готовые тексты базы."""
    if not text_ids or len(text_ids) > MAX_TFIDF_TEXTS:
        return {"texts": [], "too_many": len(text_ids) > MAX_TFIDF_TEXTS, "n_docs": 0}
    pos_set = POS_GROUPS.get(pos, POS_GROUPS["content"])[1]
    q = select(TokenRow.text_id, TokenRow.lemma_norm, func.count()).where(TokenRow.text_id.in_(text_ids))
    q = q.where(TokenRow.upos.in_(pos_set)) if pos_set else q.where(TokenRow.upos != "PUNCT")
    per_text: dict[int, Counter] = defaultdict(Counter)
    for tid, lemma, c in session.execute(q.group_by(TokenRow.text_id, TokenRow.lemma_norm)):
        if lemma and lemma not in stop and any(ch.isalpha() for ch in lemma) and c >= 2:
            per_text[tid][lemma] = c
    lemmas = set().union(*per_text.values()) if per_text else set()
    all_ids = A.corpus_text_ids(session, None)
    df = {}
    if lemmas:
        df = dict(session.execute(
            select(TokenRow.lemma_norm, func.count(func.distinct(TokenRow.text_id)))
            .where(TokenRow.lemma_norm.in_(sorted(lemmas)), TokenRow.text_id.in_(all_ids))
            .group_by(TokenRow.lemma_norm)
        ).all())
    meta = {t.id: t for t in session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.n_words)
                                              .where(TextDoc.id.in_(text_ids)))}
    out = []
    for tid in text_ids:
        t = meta.get(tid)
        if t is None:
            continue
        doc_len = t.n_words or 1
        scored = sorted(((tf_idf(c, doc_len, df.get(lemma, 1), len(all_ids)), lemma, c) for lemma, c in per_text[tid].items()),
                        reverse=True)[:top]
        out.append({"text_id": tid, "title": t.title, "author": t.author or "",
                    "words": [{"lemma": lemma, "count": c, "score": round(score * 1000, 3)} for score, lemma, c in scored]})
    return {"texts": out, "too_many": False, "n_docs": len(all_ids)}
