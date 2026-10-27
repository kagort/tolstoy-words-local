"""Семантика на векторах navec (входят в пакет natasha, 250 тыс. слов, 300 измерений, ~26 МБ; работает на CPU).

- похожие слова (кандидаты для пополнения словаря) — только те, что встречаются в выборке;
- смысловая карта словаря: проекция на плоскость и группы близких по смыслу слов;
- употребления (значения) слова: кластеры контекстов его вхождений;
- темы выборки: NMF по фрагментам текстов.

Всё считается на numpy без дополнительных библиотек.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import DictionaryEntry, Hit, SentenceRow, TextDoc, TokenRow
from app.nlp.parsers import _morph
from app.nlp.tree import norm
from app.services import analytics as A
from app.services.entities import embedding

CONTENT_POS = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")


# ------------------------------------------------------------------ векторы


def vector(word: str):
    v = embedding().get(norm(word))
    return None if v is None else np.asarray(v, dtype=np.float32)


@lru_cache(maxsize=1)
def _pq_norms() -> np.ndarray:
    pq = embedding().pq
    table = (pq.codes.astype(np.float32) ** 2).sum(-1)  # (qdim, centroids)
    out = np.empty(pq.vectors, dtype=np.float32)
    for s in range(0, pq.vectors, 50_000):
        idx = pq.indexes[s : s + 50_000]
        out[s : s + 50_000] = np.sqrt(table[np.arange(pq.qdim), idx].sum(1))
    return np.maximum(out, 1e-6)


def nearest(q: np.ndarray, topn: int = 500) -> list[tuple[str, float]]:
    """Ближайшие к вектору слова navec по косинусу — прямо по квантованным кодам, без распаковки 300 МБ."""
    pq = embedding().pq
    qs = q.reshape(pq.qdim, -1)
    table = np.einsum("kcd,kd->kc", pq.codes.astype(np.float32), qs)  # (qdim, centroids)
    dots = np.empty(pq.vectors, dtype=np.float32)
    for s in range(0, pq.vectors, 50_000):
        idx = pq.indexes[s : s + 50_000]
        dots[s : s + 50_000] = table[np.arange(pq.qdim), idx].sum(1)
    sims = dots / (_pq_norms() * (np.linalg.norm(q) or 1))
    top = np.argpartition(-sims, topn)[:topn]
    top = top[np.argsort(-sims[top])]
    words = embedding().vocab.words
    return [(words[i], float(sims[i])) for i in top]


def _is_lemma(word: str) -> bool:
    if len(word) < 3 or not all("а" <= ch <= "я" or ch in "ё-" for ch in word):
        return False
    p = _morph().parse(word)
    return bool(p) and p[0].normal_form == word and p[0].tag.POS not in ("PREP", "CONJ", "PRCL", "NPRO", "INTJ")


def similar_words(session: Session, words: list[str], text_ids: list[int], exclude: set[str] | None = None,
                  topn: int = 40, only_in_corpus: bool = True) -> dict:
    """Слова, близкие по смыслу к набору слов (центроид векторов); с частотой в выборке."""
    vecs = [v for v in (vector(w) for w in words) if v is not None]
    missing = [w for w in words if vector(w) is None]
    if not vecs:
        return {"items": [], "missing": missing, "used": []}
    q = np.mean([v / (np.linalg.norm(v) or 1) for v in vecs], axis=0)
    exclude = {norm(w) for w in (exclude or set())} | {norm(w) for w in words}
    cand = [(w, s) for w, s in nearest(q, 1500) if w not in exclude and _is_lemma(w)]
    counts = {}
    if cand and text_ids:
        counts = dict(session.execute(
            select(TokenRow.lemma_norm, func.count()).where(TokenRow.lemma_norm.in_([w for w, _ in cand]), TokenRow.text_id.in_(text_ids))
            .group_by(TokenRow.lemma_norm)).all())
    items = [{"word": w, "sim": round(s, 3), "count": counts.get(w, 0)} for w, s in cand
             if counts.get(w, 0) > 0 or not only_in_corpus]
    return {"items": items[:topn], "missing": missing, "used": [w for w in words if vector(w) is not None]}


# ------------------------------------------------------------------ кластеризация и проекция


def kmeans(x: np.ndarray, k: int, iters: int = 60, seed: int = 11) -> np.ndarray:
    """K-средних (k-means++) по косинусной близости; детерминированно."""
    n = len(x)
    k = max(1, min(k, n))
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    rng = random.Random(seed)
    centers = [xn[rng.randrange(n)]]
    for _ in range(1, k):
        d = np.min([1 - xn @ c for c in centers], axis=0).clip(0)
        total = d.sum()
        if total <= 0:
            centers.append(xn[rng.randrange(n)])
            continue
        r, acc = rng.random() * total, 0.0
        for i, di in enumerate(d):
            acc += di
            if acc >= r:
                centers.append(xn[i])
                break
    c = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(xn @ c.T, axis=1)
        if np.array_equal(new, labels) and _:
            break
        labels = new
        for j in range(k):
            m = xn[labels == j]
            if len(m):
                v = m.mean(0)
                c[j] = v / (np.linalg.norm(v) or 1)
    return labels


def pca2(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros((len(x), 2))
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    p = xc @ vt[:2].T
    return p / (np.abs(p).max() or 1)


def dictionary_map(session: Session, dictionary_id: int, k: int | None = None) -> dict:
    """Смысловая карта словаря: точки на плоскости (PCA векторов) и группы близких слов (k-средних)."""
    entries = list(session.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dictionary_id)
                                   .order_by(DictionaryEntry.surface)))
    have, vecs, missing = [], [], []
    for e in entries:
        v = vector(e.lemma) if vector(e.lemma) is not None else vector(e.surface)
        if v is None:
            missing.append(e.surface)
        else:
            have.append(e)
            vecs.append(v)
    if len(have) < 2:
        return {"points": [], "groups": [], "missing": missing, "k": 0}
    x = np.array(vecs)
    k = k or max(2, min(8, round(math.sqrt(len(have) / 2))))
    labels = kmeans(x, k)
    xy = pca2(x)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    groups = []
    for j in sorted(set(labels.tolist())):
        members = [i for i in range(len(have)) if labels[i] == j]
        centroid = xn[members].mean(0)
        members.sort(key=lambda i: -float(xn[i] @ centroid))
        groups.append({"id": j + 1, "words": [have[i].surface for i in members],
                       "cohesion": round(float(np.mean([xn[i] @ centroid for i in members]) / (np.linalg.norm(centroid) or 1)), 3)})
    points = [{"word": e.surface, "entry_id": e.id, "category": e.category or "", "group": int(labels[i]) + 1,
               "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4)} for i, e in enumerate(have)]
    return {"points": points, "groups": groups, "missing": missing, "k": k}


# ------------------------------------------------------------------ употребления слова


def senses(session: Session, entry_id: int, text_ids: list[int], k: int = 3, window: int = 5,
           stop: frozenset[str] = frozenset(), max_hits: int = 3000) -> dict:
    """Кластеры контекстов вхождений: контекст — средний вектор знаменательных слов в окне ±window."""
    hits = session.execute(select(Hit.id, Hit.sentence_id, Hit.token_idx)
                           .where(Hit.entry_id == entry_id, Hit.text_id.in_(text_ids or [-1]), A.valid())
                           .order_by(Hit.id)).all()
    if len(hits) > max_hits:
        hits = random.Random(5).sample(hits, max_hits)
    by_sent = defaultdict(list)
    for hid, sid, idx in hits:
        by_sent[sid].append((hid, idx))
    toks = defaultdict(list)
    sids = list(by_sent)
    for s0 in range(0, len(sids), 2000):
        for sid, idx, lemma, upos in session.execute(
            select(TokenRow.sentence_id, TokenRow.idx, TokenRow.lemma_norm, TokenRow.upos)
            .where(TokenRow.sentence_id.in_(sids[s0 : s0 + 2000]), TokenRow.upos != "PUNCT").order_by(TokenRow.sentence_id, TokenRow.idx)
        ):
            toks[sid].append((idx, lemma, upos))
    rows, vecs, contexts = [], [], []
    for sid, items in by_sent.items():
        words = toks.get(sid, [])
        pos = {idx: p for p, (idx, _, _) in enumerate(words)}
        for hid, idx in items:
            p = pos.get(idx)
            if p is None:
                continue
            ctx = [w[1] for q, w in enumerate(words) if q != p and abs(q - p) <= window and w[2] in CONTENT_POS and w[1] not in stop]
            vs = [v / (np.linalg.norm(v) or 1) for v in (vector(c) for c in ctx) if v is not None]
            if not vs:
                continue
            rows.append((hid, sid))
            vecs.append(np.mean(vs, axis=0))
            contexts.append(ctx)
    if len(rows) < max(4, k):
        return {"clusters": [], "points": [], "n": len(rows), "k": k}
    x = np.array(vecs)
    labels = kmeans(x, k)
    xy = pca2(x)
    overall = Counter(c for ctx in contexts for c in set(ctx))
    texts = dict(session.execute(select(SentenceRow.id, SentenceRow.text).where(SentenceRow.id.in_({sid for _, sid in rows}))).all())
    clusters = []
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    for j in sorted(set(labels.tolist())):
        members = [i for i in range(len(rows)) if labels[i] == j]
        cnt = Counter(c for i in members for c in set(contexts[i]))
        # характерные слова контекста: частые в кластере и относительно редкие в других кластерах
        score = {w: c * math.log((1 + len(rows)) / (1 + overall[w])) * (c / overall[w]) for w, c in cnt.items() if c >= 2}
        top = sorted(score, key=lambda w: -score[w])[:12]
        centroid = xn[members].mean(0)
        members.sort(key=lambda i: -float(xn[i] @ centroid))
        clusters.append({"id": j + 1, "size": len(members), "share": round(100 * len(members) / len(rows), 1), "words": top,
                         "examples": [{"sentence_id": rows[i][1], "text": texts.get(rows[i][1], "")} for i in members[:4]]})
    clusters.sort(key=lambda c: -c["size"])
    points = [{"x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "cluster": int(labels[i]) + 1,
               "sentence_id": rows[i][1], "snippet": " ".join(contexts[i][:8])} for i in range(len(rows))]
    return {"clusters": clusters, "points": points, "n": len(rows), "k": k}


# ------------------------------------------------------------------ темы (NMF)


def topics(session: Session, text_ids: list[int], k: int = 8, chunk_words: int = 1000, stop: frozenset[str] = frozenset(),
           vocab_size: int = 3000, iters: int = 150, max_chunks: int = 2000) -> dict:
    """Тематическая модель NMF по фрагментам (~chunk_words слов) текстов выборки (TF-IDF знаменательных лемм)."""
    rows = session.execute(select(TokenRow.text_id, TokenRow.sentence_id, TokenRow.lemma_norm)
                           .where(TokenRow.text_id.in_(text_ids or [-1]), TokenRow.upos.in_(CONTENT_POS))
                           .order_by(TokenRow.text_id, TokenRow.sentence_id, TokenRow.idx)).all()
    chunks, owners, cur, cur_text = [], [], [], None
    for tid, _, lemma in rows:
        if tid != cur_text and cur:
            chunks.append(cur)
            owners.append(cur_text)
            cur = []
        cur_text = tid
        if lemma and lemma not in stop and len(lemma) > 2 and any(ch.isalpha() for ch in lemma):
            cur.append(lemma)
        if len(cur) >= chunk_words // 2:  # знаменательных ~половина слов
            chunks.append(cur)
            owners.append(tid)
            cur = []
    if cur:
        chunks.append(cur)
        owners.append(cur_text)
    if len(chunks) > max_chunks:
        step = len(chunks) / max_chunks
        keep = [int(i * step) for i in range(max_chunks)]
        chunks, owners = [chunks[i] for i in keep], [owners[i] for i in keep]
    if len(chunks) < max(3, k):
        return {"topics": [], "by_text": [], "n_chunks": len(chunks), "k": k}
    df = Counter(w for c in chunks for w in set(c))
    max_df = 0.5 * len(chunks)
    vocab = [w for w, d in df.most_common() if 2 <= d <= max_df][:vocab_size]
    vi = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(chunks), len(vocab)), dtype=np.float32)
    for r, c in enumerate(chunks):
        for w, n in Counter(c).items():
            if w in vi:
                m[r, vi[w]] = n
    idf = np.log((1 + len(chunks)) / (1 + np.array([df[w] for w in vocab], dtype=np.float32))) + 1
    m = m * idf
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(3)
    w_ = rng.random((m.shape[0], k), dtype=np.float32) + 0.1
    h_ = rng.random((k, m.shape[1]), dtype=np.float32) + 0.1
    for _ in range(iters):  # мультипликативные обновления Ли — Сына
        h_ *= (w_.T @ m) / np.maximum(w_.T @ w_ @ h_, 1e-9)
        w_ *= (m @ h_.T) / np.maximum(w_ @ h_ @ h_.T, 1e-9)
    weight = w_.sum(0)
    order = np.argsort(-weight)
    topics_out = []
    for rank, j in enumerate(order, start=1):
        top = np.argsort(-h_[j])[:12]
        topics_out.append({"id": rank, "words": [vocab[i] for i in top], "share": round(float(100 * weight[j] / weight.sum()), 1)})
    from app.services.labels import text_labels

    titles = {t.id: text_labels(t.title, t.author, None, t.meta)[0] for t in session.execute(  # подпись строки графика
        select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.meta).where(TextDoc.id.in_(set(owners))))}
    by_text = []
    wn = w_ / np.maximum(w_.sum(1, keepdims=True), 1e-9)
    for tid in dict.fromkeys(owners):
        idx = [i for i, o in enumerate(owners) if o == tid]
        dist = wn[idx].mean(0)[order]
        by_text.append({"text_id": tid, "title": titles.get(tid, str(tid)), "chunks": len(idx), "dist": [round(float(100 * v), 1) for v in dist]})
    timeline = None
    if len(by_text) == 1:  # один текст — как меняются темы от начала к концу
        timeline = [[round(float(100 * v), 1) for v in wn[i][order]] for i in range(len(owners))]
    return {"topics": topics_out, "by_text": by_text, "timeline": timeline, "n_chunks": len(chunks), "k": k,
            "vocab": len(vocab)}
