"""Сводный отчёт по выбранным словам словаря и выборке текстов (вкладка «Отчёт»)."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Hit, TextDoc
from app.nlp.syntax import role_label_ud
from app.services import analytics as A

REPORT_RELATIONS = ("attribute", "head", "subject", "object", "predicate", "governor", "subject_of_pred", "adverbial")
MAX_TEXTS_FOR_TEXT_CHART = 40


def default_words(session: Session, dictionary_id: int, text_ids: list[int], n: int = 10) -> list[int]:
    return [r["entry_id"] for r in A.frequency(session, dictionary_id, text_ids)["rows"] if r["count"]][:n]


def build_report(session: Session, dictionary_id: int, text_ids: list[int], entry_ids: list[int], *,
                 step: int = 10, ids_b: list[int] | None = None, examples: int = 3, top: int = 8,
                 stop: frozenset[str] = frozenset()) -> dict:
    freq_all = A.frequency(session, dictionary_id, text_ids)
    by_id = {r["entry_id"]: r for r in freq_all["rows"]}
    words = [by_id[e] for e in entry_ids if e in by_id]
    ids = [w["entry_id"] for w in words]

    # распределение по группам: показываем только осмысленные группировки
    n_texts = len(text_ids)
    facets = A.facets(session, text_ids)
    dims = ["author"] if len(facets["authors"]) > 1 else []
    if len(facets["genres"]) > 1:
        dims.append("genre")
    if facets["year_min"] and facets["year_max"] and facets["year_max"] > facets["year_min"]:
        dims.append("period")
    stypes = session.scalar(select(func.count(func.distinct(TextDoc.source_type))).where(TextDoc.id.in_(text_ids or [-1]))) or 0
    if stypes > 1:
        dims.append("source_type")
    if 1 < n_texts <= MAX_TEXTS_FOR_TEXT_CHART:
        dims.append("text")
    overviews = {d: A.overview(session, dictionary_id, text_ids, ids, d, step) for d in dims} if ids else {}

    # роли и связи каждого слова, примеры
    cards = []
    for w in words:
        prof = A.word_profile(session, w["entry_id"], text_ids, top=top, stop=stop)
        rels = {rt: prof["relations"][rt] for rt in REPORT_RELATIONS if rt in prof["relations"]}
        conc = A.concordance(session, dictionary_id, text_ids, entry_id=w["entry_id"], per_page=examples * 3, context_chars=120)
        seen, picked = set(), []  # одно предложение — один пример, даже если слово в нём дважды
        for item in conc["items"]:
            if item["sentence_id"] not in seen and len(picked) < examples:
                seen.add(item["sentence_id"])
                picked.append(item)
        conc["items"] = picked
        top_role = prof["roles"][0] if prof["roles"] else None
        cards.append({**w, "roles": prof["roles"], "relations": rels, "examples": conc["items"],
                      "top_role": top_role["label"] if top_role else "",
                      "top_role_pct": round(100 * top_role["count"] / prof["total"], 1) if top_role and prof["total"] else 0})

    # роли: матрица слово × роль (в % вхождений слова)
    role_keys: list[str] = []
    for c in cards:
        for r in c["roles"]:
            if r["role"] not in role_keys:
                role_keys.append(r["role"])
    role_matrix = {
        "roles": [{"role": r, "label": role_label_ud(r)} for r in role_keys],
        "words": [{"word": c["word"], "pct": [round(100 * next((r["count"] for r in c["roles"] if r["role"] == k), 0) / c["count"], 1)
                                               if c["count"] else 0 for k in role_keys]} for c in cards],
    }

    structure = A.sentence_structure(session, dictionary_id, text_ids, entry_ids=ids) if ids else None
    compare = None
    if ids_b is not None:
        cmp = A.compare_frequency(session, dictionary_id, text_ids, ids_b)
        compare = {**cmp, "rows": [r for r in cmp["rows"] if r["entry_id"] in set(ids)]}

    texts_with = session.scalar(
        select(func.count(func.distinct(Hit.text_id))).where(Hit.dictionary_id == dictionary_id, Hit.entry_id.in_(ids or [-1]),
                                                          Hit.text_id.in_(text_ids or [-1]), A.VALID)) or 0
    total = sum(w["count"] for w in words)
    return {
        "words": words, "cards": cards, "overviews": overviews, "role_matrix": role_matrix, "structure": structure,
        "compare": compare, "corpus_words": freq_all["corpus_words"], "n_texts": n_texts, "texts_with": texts_with,
        "total": total, "total_ipm": round(A.ipm(total, freq_all["corpus_words"]), 2) if freq_all["corpus_words"] else 0,
        "step": step,
    }
