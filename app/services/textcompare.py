"""Панель сравнения 2–5 текстов: метрики портрета рядом, лепестковая диаграмма, общая и особая лексика,
ключевые слова каждого текста против остальных, стилометрическое расстояние."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TextDoc
from app.services.keywords import keywords
from app.services.labels import text_labels
from app.services.textstats import get_stats

METRICS = [
    ("size", "words", "Словоупотреблений", False),
    ("lexical", "mattr", "MATTR (лексическое разнообразие)", True),
    ("lexical", "yule_k", "K Юла (повторяемость лексики)", True),
    ("lexical", "density", "Лексическая плотность, %", True),
    ("lexical", "hapax_pct", "Hapax, % лемм", True),
    ("size", "avg_word_len", "Средняя длина слова, букв", True),
    ("syntax", "avg_len", "Средняя длина предложения", True),
    ("syntax", "stdev_len", "Разброс длины предложений", True),
    ("syntax", "mdd", "Средняя дистанция зависимости (MDD)", True),
    ("syntax", "avg_depth", "Глубина дерева", True),
    ("syntax", "avg_clauses", "Клауз на предложение", True),
    ("syntax", "direct_speech", "% прямой речи", True),
    ("syntax", "participial", "% с причастными оборотами", True),
    ("syntax", "gerund", "% с деепричастиями", True),
    ("morphology", "noun_verb", "Существительных на глагол", True),
    ("morphology", "adj_noun", "Прилагательных на существительное", True),
    ("morphology", "pron_share", "Местоимения, %", True),
    ("readability", "flesch_oborneva", "Удобочитаемость (Флеш — Оборнева)", True),
]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

def compare(session: Session, text_ids: list[int], stop: frozenset[str] = frozenset()) -> dict:
    meta = {t.id: t for t in session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from)
                                              .where(TextDoc.id.in_(text_ids or [-1])))}
    ids = [t for t in text_ids if t in meta]
    stats = {t: get_stats(session, t) or {} for t in ids}
    texts = [{"id": t, "title": meta[t].title, "author": meta[t].author or "", "year": meta[t].year_from} for t in ids]
    rows = []
    for sec, key, label, radar in METRICS:
        vals = [(stats[t].get(sec) or {}).get(key) for t in ids]
        rows.append({"label": label, "values": vals, "radar": radar})
    radar_rows = [r for r in rows if r["radar"] and all(v is not None for v in r["values"])]
    radar = {"labels": [r["label"].split(" (")[0].replace("Средняя ", "").replace("Средний ", "") for r in radar_rows], "series": []}
    for i, t in enumerate(texts):
        norm_vals = []
        for r in radar_rows:
            lo, hi = min(r["values"]), max(r["values"])
            norm_vals.append(round((r["values"][i] - lo) / (hi - lo), 3) if hi > lo else 0.5)
        radar["series"].append({"title": t["title"], "label": text_labels(t["title"], t["author"], None)[0], "values": norm_vals})
    # лексика: частотные знаменательные леммы из портрета (запас pool)
    pools = {}
    for t in ids:
        pool = (stats[t].get("frequency") or {}).get("pool") or []
        pools[t] = [x["lemma"] for x in pool if x["pos"] in CONTENT and x["lemma"] not in stop][:300]
    sets = {t: set(v) for t, v in pools.items()}
    jac = [[round(len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1), 3) for b in ids] for a in ids]
    common = sorted(set.intersection(*sets.values()), key=lambda w: pools[ids[0]].index(w))[:40] if sets and all(sets.values()) else []
    others_union = {t: set().union(*(sets[o] for o in ids if o != t)) if len(ids) > 1 else set() for t in ids}
    unique = {t: [w for w in pools[t] if w not in others_union[t]][:25] for t in ids}
    kw = {}
    if len(ids) > 1:
        for t in ids:
            res = keywords(session, [t], [o for o in ids if o != t], min_count=5, stop=stop, top=20)
            kw[t] = [r["lemma"] for r in res["positive"] if r["significant"]][:20]
    delta = None
    if len(ids) >= 2:
        from app.services.stylometry import _units, distance_matrix, zscores

        units, words = _units(session, ids, "text", 0)
        if len(units) >= 2:
            z, _ = zscores(words, 100)
            delta = [[round(float(v), 3) for v in row] for row in distance_matrix(z, "cosine")]
    return {"texts": texts, "rows": rows, "radar": radar, "jaccard": jac, "common": common,
            "unique": [unique[t] for t in ids], "keywords": [kw.get(t, []) for t in ids], "delta": delta}
