"""Карта словаря по тексту («где самые пахучие места») и лексическая дисперсия."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DictionaryEntry, Hit, SentenceRow, TextDoc
from app.services import analytics as A
from app.services.stats import gries_dp, ipm, juilland_d

SEGMENT_CHOICES = (10, 20, 50, 100)


def text_map(session: Session, text_id: int, dictionary_id: int, entry_ids: list[int] | None = None,
             segments: int = 50, top: int = 15, by: str = "word") -> dict:
    """Плотность вхождений по равным (по числу слов) отрезкам текста.

    Возвращает: строки «слово × отрезок» (вхождения и ipm), строку «все слова словаря», строки категорий,
    дисперсию каждого слова (DP Грайса, D Жуйяна, число отрезков со словом) и точки для графика рассеяния.
    """
    segments = segments if segments in SEGMENT_CHOICES else 50
    doc = session.get(TextDoc, text_id)
    sents = session.execute(select(SentenceRow.id, SentenceRow.idx, SentenceRow.n_words)
                            .where(SentenceRow.text_id == text_id).order_by(SentenceRow.idx)).all()
    total = sum(s.n_words or 0 for s in sents) or 1
    start_of, seg_of = {}, {}
    seg_words = [0] * segments
    seg_from, seg_to = [None] * segments, [None] * segments
    offset = 0
    for s in sents:
        seg = min(segments - 1, offset * segments // total)
        start_of[s.id], seg_of[s.id] = offset, seg
        seg_words[seg] += s.n_words or 0
        seg_from[seg] = s.idx if seg_from[seg] is None else seg_from[seg]
        seg_to[seg] = s.idx
        offset += s.n_words or 0

    entries = {e.id: e for e in session.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dictionary_id))}
    hits = session.execute(select(Hit.entry_id, Hit.sentence_id, Hit.token_idx)
                           .where(Hit.dictionary_id == dictionary_id, Hit.text_id == text_id, A.valid())).all()
    per_entry: dict[int, list[int]] = defaultdict(lambda: [0] * segments)
    points = defaultdict(list)
    for eid, sid, tidx in hits:
        if sid not in seg_of:
            continue
        per_entry[eid][seg_of[sid]] += 1
        points[eid].append(round(100 * (start_of[sid] + tidx) / total, 2))
    if by == "cat":  # строки — категории словаря (выбранных слов или все): вхождения слов категории складываются
        cat_of = {e: (entries[e].category or "без категории") for e in per_entry}
        scope = [e for e in per_entry if not entry_ids or e in set(entry_ids)]
        per_cat: dict[str, list[int]] = defaultdict(lambda: [0] * segments)
        cat_points: dict[str, list[float]] = defaultdict(list)
        for e in scope:
            for k in range(segments):
                per_cat[cat_of[e]][k] += per_entry[e][k]
            cat_points[cat_of[e]] += points[e]
        per_entry, points = per_cat, {c: sorted(v) for c, v in cat_points.items()}
        entries = {c: SimpleNamespace(surface=c, category=None) for c in per_cat}  # подпись строки = категория
        chosen = sorted(per_cat, key=lambda c: -sum(per_cat[c]))[: max(top, 30)]
    else:
        chosen = [e for e in (entry_ids or []) if e in entries]
        if not chosen:
            chosen = sorted(per_entry, key=lambda e: -sum(per_entry[e]))[:top]

    def row(label, counts, kind="word"):
        return {"label": label, "kind": kind, "count": counts, "total": sum(counts),
                "ipm": [round(ipm(c, w), 1) if w else 0 for c, w in zip(counts, seg_words)],
                "dp": round(gries_dp(counts, seg_words), 3) if sum(counts) else None,
                "d": round(juilland_d(counts), 3) if sum(counts) and juilland_d(counts) is not None else None,
                "range": sum(1 for c in counts if c)}

    rows = [row(entries[e].surface, per_entry.get(e, [0] * segments)) for e in chosen]
    all_counts = [sum(per_entry[e][k] for e in per_entry) for k in range(segments)]
    summary = [row("все слова словаря", all_counts, "all")]
    cats = defaultdict(lambda: [0] * segments)
    from app.models import Dictionary
    from app.services.dictionaries import no_categories

    flat = no_categories(session.get(Dictionary, dictionary_id))  # словарь «без категорий» — без строк категорий
    for e, counts in per_entry.items():
        if entries[e].category and not flat:
            for k in range(segments):
                cats[entries[e].category][k] += counts[k]
    summary += [row(f"категория «{c}»", v, "category") for c, v in sorted(cats.items())]
    peaks = sorted(range(segments), key=lambda k: -summary[0]["ipm"][k])[:5]
    return {
        "text_id": text_id, "title": doc.title if doc else "", "segments": segments, "seg_words": seg_words,
        "seg_from": seg_from, "seg_to": seg_to, "rows": rows, "summary": summary,
        "points": [{"label": entries[e].surface, "x": points.get(e, [])} for e in chosen],
        "peaks": [{"segment": k + 1, "from": seg_from[k], "to": seg_to[k], "ipm": summary[0]["ipm"][k],
                   "count": summary[0]["count"][k]} for k in peaks if summary[0]["count"][k]],
        "n_words": total, "by": by,
    }


def passage(session: Session, text_id: int, from_idx: int, to_idx: int, dictionary_id: int | None,
            entry_ids: list[int] | None = None, limit: int = 80) -> list[dict]:
    """Предложения отрезка текста с подсветкой вхождений словаря (для щелчка по карте)."""
    to_idx = min(to_idx, from_idx + limit - 1)
    sents = session.execute(select(SentenceRow.id, SentenceRow.idx, SentenceRow.paragraph_idx, SentenceRow.text)
                            .where(SentenceRow.text_id == text_id, SentenceRow.idx.between(from_idx, to_idx))
                            .order_by(SentenceRow.idx)).all()
    marks = defaultdict(list)
    if dictionary_id and sents:
        from app.models import TokenRow

        q = (select(Hit.sentence_id, TokenRow.start_char, TokenRow.end_char, Hit.entry_id)
             .join(TokenRow, TokenRow.id == Hit.token_id)
             .where(Hit.dictionary_id == dictionary_id, Hit.sentence_id.in_([s.id for s in sents]), A.valid()))
        if entry_ids:
            q = q.where(Hit.entry_id.in_(entry_ids))
        for sid, st, en, eid in session.execute(q):
            marks[sid].append((st, en))
    return [{"id": s.id, "idx": s.idx, "paragraph": s.paragraph_idx, "text": s.text, "marks": sorted(marks.get(s.id, []))}
            for s in sents]
