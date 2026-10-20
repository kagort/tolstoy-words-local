"""Стилометрия: Delta Барроуза и Cosine Delta по самым частотным словам (MFW), кластеризация и атрибуция.

Единицы сравнения — тексты или фрагменты текстов по N слов (фрагменты позволяют проверять однородность текста и
работать с малым числом текстов). Признаки — относительные частоты самых частых словоформ выборки (в основном
служебные слова: их употребление автор почти не контролирует). Стоп-слова здесь НЕ применяются.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TextDoc, TokenRow

MFW_CHOICES = (50, 100, 200, 300, 500)
DISTANCES = {"cosine": "Cosine Delta (рекомендуется)", "burrows": "Delta Барроуза"}
MIN_WORDS = 5000  # меньше — результаты ненадёжны


def _units(session: Session, text_ids: list[int], unit: str, chunk: int) -> tuple[list[dict], list[list[str]]]:
    meta = {t.id: t for t in session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from, TextDoc.source_type)
                                              .where(TextDoc.id.in_(text_ids or [-1])))}
    units, words = [], []
    for tid in text_ids:
        t = meta.get(tid)
        if t is None:
            continue
        seq = list(session.scalars(select(TokenRow.form_norm).where(TokenRow.text_id == tid, TokenRow.upos != "PUNCT")
                                   .order_by(TokenRow.sentence_id, TokenRow.idx)))
        seq = [w for w in seq if any(ch.isalpha() for ch in w)]
        base = {"text_id": tid, "title": t.title, "author": t.author or "—", "year": t.year_from, "stype": t.source_type}
        if unit == "chunk" and len(seq) >= chunk:
            for k in range(len(seq) // chunk):
                units.append({**base, "label": f"{t.title} [{k + 1}]", "part": k + 1, "n_words": chunk})
                words.append(seq[k * chunk : (k + 1) * chunk])
        else:
            units.append({**base, "label": t.title, "part": None, "n_words": len(seq)})
            words.append(seq)
    return units, words


def zscores(words: list[list[str]], mfw: int) -> tuple[np.ndarray, list[str]]:
    total = Counter()
    for ws in words:
        total.update(ws)
    features = [w for w, _ in total.most_common(mfw)]
    freq = np.zeros((len(words), len(features)))
    for i, ws in enumerate(words):
        c = Counter(ws)
        n = max(len(ws), 1)
        freq[i] = [c.get(f, 0) / n for f in features]
    mu, sd = freq.mean(0), freq.std(0)
    sd[sd == 0] = 1
    return (freq - mu) / sd, features


def distance_matrix(z: np.ndarray, kind: str = "cosine") -> np.ndarray:
    if kind == "burrows":
        return np.abs(z[:, None, :] - z[None, :, :]).mean(-1)
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    return np.clip(1 - zn @ zn.T, 0, 2)


def upgma(d: np.ndarray) -> list[tuple[int, int, float, int]]:
    """Иерархическая кластеризация (средняя связь). Возвращает слияния (a, b, высота, размер) как в scipy.linkage."""
    n = len(d)
    clusters = {i: [i] for i in range(n)}
    dist = {(i, j): d[i, j] for i in range(n) for j in range(i + 1, n)}
    merges, next_id = [], n
    while len(clusters) > 1:
        (a, b), h = min(dist.items(), key=lambda kv: kv[1])
        members = clusters.pop(a) + clusters.pop(b)
        dist = {k: v for k, v in dist.items() if a not in k and b not in k}
        for c, mc in clusters.items():
            dist[(min(c, next_id), max(c, next_id))] = float(np.mean([d[i, j] for i in members for j in mc]))
        clusters[next_id] = members
        merges.append((a, b, float(h), len(members)))
        next_id += 1
    return merges


def dendrogram_coords(merges: list[tuple[int, int, float, int]], n: int) -> dict:
    """Координаты линий дендрограммы (горизонтальная: листья по вертикали, высота слияния — по горизонтали)."""
    if n == 1:
        return {"order": [0], "segments": []}
    children = {n + k: (a, b) for k, (a, b, _, _) in enumerate(merges)}
    height = {n + k: h for k, (_, _, h, _) in enumerate(merges)}
    order = []

    def walk(node):
        if node < n:
            order.append(node)
        else:
            walk(children[node][0])
            walk(children[node][1])

    walk(n + len(merges) - 1)
    ypos = {leaf: i for i, leaf in enumerate(order)}
    segs = []

    def place(node):
        if node < n:
            return ypos[node], 0.0
        a, b = children[node]
        ya, ha = place(a)
        yb, hb = place(b)
        h = height[node]
        segs.extend([[ha, ya, h, ya], [hb, yb, h, yb], [h, ya, h, yb]])
        return (ya + yb) / 2, h

    place(n + len(merges) - 1)
    return {"order": order, "segments": segs}


def mds2(d: np.ndarray) -> np.ndarray:
    """Классическое многомерное шкалирование расстояний в 2 измерения."""
    n = len(d)
    if n < 3:
        return np.array([[i, 0.0] for i in range(n)])
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    idx = np.argsort(vals)[::-1][:2]
    xy = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))
    return xy / (np.abs(xy).max() or 1)


def analyze(session: Session, text_ids: list[int], *, mfw: int = 100, kind: str = "cosine", unit: str = "text",
            chunk: int = 5000, target: int | None = None) -> dict:
    """Расстояния между текстами (фрагментами), дендрограмма, карта и, если задан target, ранжирование авторов."""
    mfw = mfw if mfw in MFW_CHOICES else 100
    kind = kind if kind in DISTANCES else "cosine"
    units, words = _units(session, text_ids, unit, chunk)
    out = {"units": units, "mfw": mfw, "kind": kind, "unit": unit, "chunk": chunk, "features": [], "small": [],
           "matrix": [], "dendro": None, "map": [], "ranking": None}
    if len(units) < 3:
        out["error"] = "Нужно хотя бы 3 текста (или фрагмента). Выберите «фрагменты» или добавьте тексты."
        return out
    z, features = zscores(words, mfw)
    d = distance_matrix(z, kind)
    out["features"] = features
    out["small"] = sorted({u["title"] for u in units if u["n_words"] < MIN_WORDS})
    out["matrix"] = [[round(float(v), 3) for v in row] for row in d]
    merges = upgma(d)
    out["dendro"] = dendrogram_coords(merges, len(units))
    out["map"] = [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in mds2(d)]
    # кто ближе всего: соседи каждой единицы
    for i, u in enumerate(units):
        nearest = sorted((j for j in range(len(units)) if j != i), key=lambda j: d[i, j])
        u["nearest"] = units[nearest[0]]["label"] if nearest else ""
        u["nearest_author"] = units[nearest[0]]["author"] if nearest else ""
    if target:
        out["ranking"] = attribute(units, d, target)
    out["purity"] = _purity(units, d)
    return out


def attribute(units: list[dict], d: np.ndarray, target: int) -> dict:
    """Ранжирование авторов для спорного текста: среднее расстояние его единиц до единиц каждого автора."""
    tgt = [i for i, u in enumerate(units) if u["text_id"] == target]
    others = [i for i, u in enumerate(units) if u["text_id"] != target]
    by_author = defaultdict(list)
    for j in others:
        by_author[units[j]["author"]].append(j)
    rows = []
    for author, js in by_author.items():
        dist = float(np.mean([d[i, j] for i in tgt for j in js])) if tgt else math.nan
        rows.append({"author": author, "distance": round(dist, 3), "units": len(js),
                     "texts": len({units[j]["text_id"] for j in js})})
    rows.sort(key=lambda r: r["distance"])
    title = units[tgt[0]]["title"] if tgt else ""
    margin = round(rows[1]["distance"] - rows[0]["distance"], 3) if len(rows) > 1 else None
    return {"target": target, "title": title, "rows": rows, "margin": margin}


def _purity(units: list[dict], d: np.ndarray) -> float | None:
    """Доля единиц, ближайший сосед которых — того же автора (проверка, «узнаёт» ли метод авторов в выборке)."""
    authors = {u["author"] for u in units}
    if len(authors) < 2:
        return None
    ok = 0
    for i, u in enumerate(units):
        j = min((k for k in range(len(units)) if k != i), key=lambda k: d[i, k])
        ok += units[j]["author"] == u["author"]
    return round(100 * ok / len(units), 1)
