"""Коллокации слова словаря: оконные (±N слов) и синтаксические, с полным набором мер ассоциации."""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Hit, TokenRelation, TokenRow
from app.nlp.syntax import rel_label_ud
from app.nlp.tree import norm
from app.services import analytics as A
from app.services.stats import MEASURES, association

CONTENT_POS = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")
POS_GROUPS = {
    "content": ("знаменательные", CONTENT_POS),
    "NOUN": ("существительные", ("NOUN", "PROPN")),
    "VERB": ("глаголы", ("VERB",)),
    "ADJ": ("прилагательные", ("ADJ",)),
    "ADV": ("наречия", ("ADV",)),
    "all": ("все части речи", None),
}
MODES = {"window": "в окне ±N слов", "syntax": "по синтаксическим связям"}


def _lemma_freq(session: Session, lemmas: set[str], text_ids: list[int]) -> dict[str, int]:
    if not lemmas or not text_ids:
        return {}
    return dict(session.execute(
        select(TokenRow.lemma_norm, func.count())
        .where(TokenRow.lemma_norm.in_(sorted(lemmas)), TokenRow.text_id.in_(text_ids))
        .group_by(TokenRow.lemma_norm)
    ).all())


def collocations(session: Session, entry_id: int, text_ids: list[int], *, mode: str = "window", window: int = 3,
                 pos: str = "content", min_count: int = 2, stop: frozenset[str] = frozenset(),
                 sort: str = "logdice", top: int = 60) -> dict:
    """Возвращает коллокаты с мерами: count, logDice, PMI, NPMI, MI³, t-score, G², ΔP в обе стороны."""
    mode = mode if mode in MODES else "window"
    window = max(1, min(int(window or 3), 10))
    pos_set = POS_GROUPS.get(pos, POS_GROUPS["content"])[1]
    sort = sort if sort in MEASURES else "logdice"
    base = {"mode": mode, "window": window, "pos": pos, "min_count": min_count, "sort": sort, "rows": [], "fx": 0,
            "n": 0, "span": 1}
    if not text_ids:
        return base
    hits = session.execute(
        select(Hit.sentence_id, Hit.token_idx).where(Hit.entry_id == entry_id, Hit.text_id.in_(text_ids), A.VALID)
    ).all()
    fx = len(hits)
    n = A.corpus_words(session, text_ids)
    base.update(fx=fx, n=n)
    if not fx:
        return base

    counts: Counter = Counter()
    pos_of: dict[tuple, Counter] = defaultdict(Counter)
    if mode == "window":
        by_sent = defaultdict(list)
        for sid, idx in hits:
            by_sent[sid].append(idx)
        sids = list(by_sent)
        for k in range(0, len(sids), 2000):
            chunk = sids[k : k + 2000]
            toks = defaultdict(list)
            for sid, idx, lemma, upos in session.execute(
                select(TokenRow.sentence_id, TokenRow.idx, TokenRow.lemma_norm, TokenRow.upos)
                .where(TokenRow.sentence_id.in_(chunk), TokenRow.upos != "PUNCT").order_by(TokenRow.sentence_id, TokenRow.idx)
            ):
                toks[sid].append((idx, lemma, upos))
            for sid in chunk:
                words = toks.get(sid, [])
                position = {idx: p for p, (idx, _, _) in enumerate(words)}
                for hidx in by_sent[sid]:
                    p = position.get(hidx)
                    if p is None:
                        continue
                    for q in range(max(0, p - window), min(len(words), p + window + 1)):
                        if q == p:
                            continue
                        _, lemma, upos = words[q]
                        key = (lemma, "")
                        counts[key] += 1
                        pos_of[key][upos] += 1
        span = 2 * window
    else:
        rel_q = (
            select(TokenRelation.rel_type, TokenRelation.related_lemma, TokenRelation.related_upos, func.count())
            .join(Hit, Hit.id == TokenRelation.hit_id)
            .where(TokenRelation.entry_id == entry_id, TokenRelation.text_id.in_(text_ids), A.VALID,
                   TokenRelation.scope.in_(list(A.DEFAULT_SCOPES)), TokenRelation.negated.is_(False),
                   TokenRelation.rel_type != "case")
            .group_by(TokenRelation.rel_type, TokenRelation.related_lemma, TokenRelation.related_upos)
        )
        for rel, lemma, upos, c in session.execute(rel_q):
            key = (norm(lemma), rel)
            counts[key] += c
            pos_of[key][upos or "X"] += c
        span = 1
    base["span"] = span

    keep = {}
    for key, c in counts.items():
        if c < min_count or not key[0] or key[0] in stop:
            continue
        main_pos = pos_of[key].most_common(1)[0][0]
        if pos_set and main_pos not in pos_set:
            continue
        keep[key] = (c, main_pos)
    fy = _lemma_freq(session, {k[0] for k in keep}, text_ids)
    rows = []
    for (lemma, rel), (c, main_pos) in keep.items():
        m = association(c, fx, fy.get(lemma, c), n, span)
        rows.append({"lemma": lemma, "rel": rel, "rel_label": rel_label_ud(rel) if rel else "", "pos": main_pos,
                     "fy": fy.get(lemma, c), **m})
    rows.sort(key=lambda r: (r.get(sort) is None, -(r.get(sort) or 0), -r["count"]))
    base["rows"] = rows[:top]
    base["total_candidates"] = len(rows)
    return base


def orbit_layout(rows: list[dict], measure: str, rings: int = 4) -> list[dict]:
    """Координаты для орбитального графа: самые сильные коллокаты — на ближней орбите."""
    import math

    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: (r.get(measure) is None, -(r.get(measure) or 0)))
    per_ring = max(1, math.ceil(len(ranked) / rings))
    out = []
    for i, r in enumerate(ranked):
        ring = i // per_ring + 1
        in_ring = ranked[(ring - 1) * per_ring : ring * per_ring]
        k = in_ring.index(r)
        angle = 2 * math.pi * (k + 0.5 * (ring % 2)) / len(in_ring)
        out.append({**r, "ring": ring, "x": round(ring * math.cos(angle), 4), "y": round(ring * math.sin(angle), 4)})
    return out
