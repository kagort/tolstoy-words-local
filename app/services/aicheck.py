"""Признаки машинного текста: объяснимый стилометрический отчёт для преподавателя.

Это НЕ детектор с вердиктом. Отчёт показывает, насколько текст похож по набору объяснимых признаков на тексты,
которые в базе помечены как «сгенерирован нейросетью», и на «литературу / естественные» тексты. Модель
(логистическая регрессия на numpy) калибруется на ваших же текстах — по фрагментам ~400 слов; её точность
на этих данных показывается честно (перекрёстная проверка). Короткие тексты (< 300 слов) не оцениваются.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import SentenceRow, TextDoc, TokenRow
from app.nlp.tree import norm, parse_feats

MIN_WORDS = 300
CHUNK_WORDS = 400
MAX_CHUNKS_PER_TEXT = 150
MARKERS_FILE = Path(__file__).resolve().parent.parent / "data" / "ai_markers_ru.txt"
PRON1 = {"я", "мы", "меня", "мне", "мной", "нас", "нам", "нами", "мой", "наш"}

FEATURES = {
    "sent_len_mean": ("Средняя длина предложения, слов", "длинные ровные предложения характерны для сгенерированного текста"),
    "sent_len_cv": ("Разнообразие длины предложений (коэф. вариации)", "человек чередует короткие и длинные фразы; у моделей длина ровнее"),
    "par_len_cv": ("Разнообразие длины абзацев", "у моделей абзацы часто одинакового размера"),
    "mattr": ("Лексическое разнообразие (MATTR, окно 100)", "насколько разнообразна лексика"),
    "markers": ("Шаблонные связки и клише, на 1000 слов", "«важно отметить», «играет ключевую роль», «уникальный», «атмосфера»…"),
    "repeat_3grams": ("Повторы сочетаний из 3 слов, %", "доля повторяющихся триграмм — шаблонность"),
    "opener_diversity": ("Разнообразие начал предложений, %", "доля разных первых слов предложений"),
    "adj_share": ("Прилагательные, % слов", "насыщенность эпитетами"),
    "noun_verb": ("Существительных на глагол", "именной (>1,5) или глагольный стиль"),
    "pron1": ("Местоимения 1-го лица, на 1000 слов", "личная позиция автора, «я», «мы»"),
    "part_ger": ("Причастия и деепричастия, на 1000 слов", "обороты «вызывая», «создавая», «наполняющий»"),
    "dashes": ("Тире, на 1000 слов", ""),
    "commas": ("Запятые, на 1000 слов", ""),
    "excl_quest": ("Восклицательные и вопросительные предложения, %", "эмоциональность, диалогичность"),
    "dialogue": ("Предложения прямой речи / диалога, %", ""),
    "word_len": ("Средняя длина слова, букв", ""),
}


@lru_cache(maxsize=1)
def markers() -> tuple[tuple[str, ...], ...]:
    out = []
    for line in MARKERS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(tuple(norm(w) for w in line.split()))
    return tuple(out)


def _mattr(lemmas: list[str], window: int = 100) -> float:
    if len(lemmas) <= window:
        return len(set(lemmas)) / max(len(lemmas), 1)
    counts = Counter(lemmas[:window])
    total = len(counts)
    vals = [total]
    for i in range(window, len(lemmas)):
        out, inn = lemmas[i - window], lemmas[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
            total -= 1
        if counts[inn] == 0:
            total += 1
        counts[inn] += 1
        vals.append(total)
    return sum(vals) / len(vals) / window


def features(sentences: list[dict]) -> dict:
    """sentences: [{"paragraph": int, "text": str, "tokens": [(form, lemma, upos, feats_dict)]}] → признаки."""
    words, lemmas, forms = [], [], []
    lens, par_words, openers = [], Counter(), []
    adj = noun = verb = pron1 = partger = dashes = commas = exq = dialogue = 0
    for s in sentences:
        ws = [t for t in s["tokens"] if t[2] != "PUNCT" and any(ch.isalpha() for ch in t[0])]
        if not ws:
            continue
        lens.append(len(ws))
        par_words[s["paragraph"]] += len(ws)
        openers.append(norm(ws[0][1]))
        txt = s["text"].strip()
        if txt.endswith(("!", "?", "!»", "?»", "?!", "!..", "?..")):
            exq += 1
        if txt.startswith(("—", "–", "-", "«", '"')):
            dialogue += 1
        for form, lemma, upos, feats in s["tokens"]:
            if upos == "PUNCT":
                dashes += form in ("—", "–")
                commas += form == ","
                continue
            if not any(ch.isalpha() for ch in form):
                continue
            words.append(form)
            lemmas.append(norm(lemma))
            forms.append(norm(form))
            adj += upos == "ADJ"
            noun += upos in ("NOUN", "PROPN")
            verb += upos == "VERB"
            pron1 += norm(form) in PRON1
            partger += feats.get("VerbForm") in ("Part", "Conv")
    n = len(words)
    if not n:
        return {}
    k = 1000 / n
    grams = [tuple(forms[i : i + 3]) for i in range(len(forms) - 2)]
    gc = Counter(grams)
    rep = sum(c for c in gc.values() if c > 1) / max(len(grams), 1)
    mk = 0
    joined = forms
    for m in markers():
        L = len(m)
        mk += sum(1 for i in range(len(joined) - L + 1) if tuple(joined[i : i + L]) == m)
    pw = list(par_words.values())

    def cv(xs):
        return statistics.pstdev(xs) / statistics.mean(xs) if len(xs) > 1 and statistics.mean(xs) else 0.0

    return {
        "n_words": n, "n_sentences": len(lens),
        "sent_len_mean": round(statistics.mean(lens), 2),
        "sent_len_cv": round(cv(lens), 3),
        "par_len_cv": round(cv(pw), 3) if len(pw) > 2 else None,
        "mattr": round(_mattr(lemmas), 4),
        "markers": round(mk * k, 2),
        "repeat_3grams": round(100 * rep, 2),
        "opener_diversity": round(100 * len(set(openers)) / len(openers), 1),
        "adj_share": round(100 * adj / n, 2),
        "noun_verb": round(noun / verb, 2) if verb else None,
        "pron1": round(pron1 * k, 2),
        "part_ger": round(partger * k, 2),
        "dashes": round(dashes * k, 2),
        "commas": round(commas * k, 2),
        "excl_quest": round(100 * exq / len(lens), 1),
        "dialogue": round(100 * dialogue / len(lens), 1),
        "word_len": round(sum(len(w) for w in words) / n, 2),
    }


# ------------------------------------------------------------------ разбор вставленного текста


def parse_raw(text: str) -> list[dict]:
    """Вставленный текст: деление на предложения (razdel), морфология Natasha, леммы pymorphy3 — без записи в базу."""
    from app.nlp.normalize import normalize_text, paragraphs
    from app.nlp.parsers import lemmatize
    from app.nlp.segment import split_paragraph
    from app.services.entities import embedding

    tagger = _morph_tagger(embedding())
    sents = []
    for p_idx, par in enumerate(paragraphs(normalize_text(text))):
        for seg in split_paragraph(par):
            sents.append({"paragraph": p_idx, "text": seg.text, "forms": [t.text for t in seg.tokens]})
    out = []
    for s, markup in zip(sents, tagger.map([s["forms"] for s in sents])):
        toks = []
        for t in markup.tokens:
            feats = dict(t.feats or {})
            toks.append((t.text, lemmatize(t.text, t.pos, feats), t.pos, feats))
        out.append({"paragraph": s["paragraph"], "text": s["text"], "tokens": toks})
    return out


_TAGGER = None


def _morph_tagger(emb):
    global _TAGGER
    if _TAGGER is None:
        from natasha import NewsMorphTagger

        _TAGGER = NewsMorphTagger(emb)
    return _TAGGER


def db_sentences(session: Session, text_id: int, limit_words: int | None = None) -> list[dict]:
    rows = session.execute(
        select(SentenceRow.id, SentenceRow.paragraph_idx, SentenceRow.text, TokenRow.form, TokenRow.lemma, TokenRow.upos, TokenRow.feats)
        .join(TokenRow, TokenRow.sentence_id == SentenceRow.id).where(SentenceRow.text_id == text_id)
        .order_by(SentenceRow.idx, TokenRow.idx)
    ).all()
    out, cur, words = [], None, 0
    for sid, par, txt, form, lemma, upos, feats in rows:
        if cur is None or cur["id"] != sid:
            if limit_words and words >= limit_words:
                break
            cur = {"id": sid, "paragraph": par, "text": txt, "tokens": []}
            out.append(cur)
        cur["tokens"].append((form, lemma, upos, parse_feats(feats)))
        words += upos != "PUNCT"
    return out


def sample_chunks(session: Session, text_id: int, max_chunks: int = MAX_CHUNKS_PER_TEXT, size: int = CHUNK_WORDS) -> list[list[dict]]:
    """Фрагменты ~size слов, взятые равномерно по всему тексту (читаются только нужные предложения)."""
    sents = session.execute(select(SentenceRow.id, SentenceRow.n_words).where(SentenceRow.text_id == text_id)
                            .order_by(SentenceRow.idx)).all()
    groups, cur, n = [], [], 0
    for sid, nw in sents:
        cur.append(sid)
        n += nw or 0
        if n >= size:
            groups.append(cur)
            cur, n = [], 0
    if len(groups) > max_chunks:
        step = len(groups) / max_chunks
        groups = [groups[int(i * step)] for i in range(max_chunks)]
    ids = [sid for g in groups for sid in g]
    by_id = {}
    for k in range(0, len(ids), 5000):
        part = ids[k : k + 5000]
        for sid, par, txt, form, lemma, upos, feats in session.execute(
            select(SentenceRow.id, SentenceRow.paragraph_idx, SentenceRow.text, TokenRow.form, TokenRow.lemma, TokenRow.upos, TokenRow.feats)
            .join(TokenRow, TokenRow.sentence_id == SentenceRow.id).where(SentenceRow.id.in_(part)).order_by(SentenceRow.id, TokenRow.idx)
        ):
            d = by_id.setdefault(sid, {"id": sid, "paragraph": par, "text": txt, "tokens": []})
            d["tokens"].append((form, lemma, upos, parse_feats(feats)))
    return [[by_id[sid] for sid in g if sid in by_id] for g in groups]


def chunks(sentences: list[dict], size: int = CHUNK_WORDS) -> list[list[dict]]:
    out, cur, n = [], [], 0
    for s in sentences:
        cur.append(s)
        n += sum(1 for t in s["tokens"] if t[2] != "PUNCT")
        if n >= size:
            out.append(cur)
            cur, n = [], 0
    if n >= size * 0.6:
        out.append(cur)
    return out


# ------------------------------------------------------------------ калибровка на текстах базы

_CALIB: dict = {}
KEYS = [k for k in FEATURES]


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r.get(k) if r.get(k) is not None else np.nan for k in KEYS] for r in rows], dtype=float)


def _logreg(x: np.ndarray, y: np.ndarray, l2: float = 0.5, iters: int = 600, lr: float = 0.2) -> tuple[np.ndarray, float]:
    w, b = np.zeros(x.shape[1]), 0.0
    pos = y.mean()
    sw = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))  # балансировка классов
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g = (p - y) * sw
        w -= lr * (x.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def calibrate(session: Session) -> dict:
    """Модель по фрагментам текстов: «сгенерирован нейросетью» (1) против остальных (0). Кэшируется до изменения базы."""
    key = tuple(session.execute(select(func.count(TextDoc.id), func.coalesce(func.sum(TextDoc.n_words), 0), func.max(TextDoc.id))
                                .where(TextDoc.status == "ready")).one())
    if _CALIB.get("key") == key:
        return _CALIB["model"]
    texts = list(session.execute(select(TextDoc.id, TextDoc.source_type, TextDoc.title).where(TextDoc.status == "ready")))
    rows, y, owner = [], [], []
    for t in texts:
        if t.source_type not in ("generated", "natural"):
            continue
        for ch in sample_chunks(session, t.id):
            f = features(ch)
            if f:
                rows.append(f)
                y.append(1 if t.source_type == "generated" else 0)
                owner.append(t.id)
    y = np.array(y)
    model = {"ok": False, "n_ai": int(y.sum()) if len(y) else 0, "n_human": int((1 - y).sum()) if len(y) else 0,
             "texts_ai": len({o for o, v in zip(owner, y) if v == 1}), "texts_human": len({o for o, v in zip(owner, y) if v == 0})}
    if model["n_ai"] < 10 or model["n_human"] < 10:
        _CALIB.update(key=key, model=model)
        return model
    x = _matrix(rows)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isnan(med), 0, med)
    x = np.where(np.isnan(x), med, x)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    xz = (x - mu) / sd
    w, b = _logreg(xz, y)
    # перекрёстная проверка по ТЕКСТАМ (фрагменты одного текста не попадают и в обучение, и в проверку)
    owners = np.array(owner)
    uniq = sorted(set(owner))
    folds = [uniq[i::5] for i in range(5)]
    correct = total = 0
    for fold in folds:
        test = np.isin(owners, fold)
        if test.all() or not test.any() or len(set(y[~test])) < 2:
            continue
        w2, b2 = _logreg(xz[~test], y[~test])
        pred = (1 / (1 + np.exp(-(xz[test] @ w2 + b2)))) >= 0.5
        correct += int((pred == y[test]).sum())
        total += int(test.sum())
    model.update(ok=True, w=w.tolist(), b=float(b), mu=mu.tolist(), sd=sd.tolist(), med=med.tolist(),
                 cv_accuracy=round(100 * correct / total, 1) if total else None, cv_n=total,
                 human_mean=x[y == 0].mean(0).tolist(), ai_mean=x[y == 1].mean(0).tolist(),
                 human_sd=x[y == 0].std(0).tolist(), ai_sd=x[y == 1].std(0).tolist())
    _CALIB.update(key=key, model=model)
    return model


# ------------------------------------------------------------------ отчёт


def report(session: Session, sentences: list[dict]) -> dict:
    f = features(sentences)
    out = {"features": f, "n_words": f.get("n_words", 0), "enough": f.get("n_words", 0) >= MIN_WORDS, "rows": [], "model": None}
    model = calibrate(session)
    out["model"] = {k: v for k, v in model.items() if k not in ("w", "mu", "sd", "med", "human_mean", "ai_mean", "human_sd", "ai_sd")}
    for i, k in enumerate(KEYS):
        label, help_ = FEATURES[k]
        row = {"key": k, "label": label, "help": help_, "value": f.get(k)}
        if model.get("ok"):
            hm, am = model["human_mean"][i], model["ai_mean"][i]
            row.update(human=round(hm, 2), ai=round(am, 2))
            v = f.get(k)
            if v is not None and abs(am - hm) > 1e-9:
                row["position"] = round(max(-0.25, min(1.25, (v - hm) / (am - hm))), 3)  # 0 — как у человека, 1 — как у модели
            z = ((v if v is not None else model["med"][i]) - model["mu"][i]) / model["sd"][i]
            row["contribution"] = round(model["w"][i] * z, 3)
        out["rows"].append(row)
    if model.get("ok") and out["enough"]:
        probs = []
        parts = chunks(sentences) or [sentences]
        for ch in parts:
            fc = features(ch)
            if not fc:
                continue
            xv = np.array([fc.get(k) if fc.get(k) is not None else model["med"][i] for i, k in enumerate(KEYS)], dtype=float)
            z = (xv - np.array(model["mu"])) / np.array(model["sd"])
            probs.append(1 / (1 + math.exp(-(float(z @ np.array(model["w"])) + model["b"]))))
        if probs:
            p = statistics.mean(probs)
            lo, hi = (min(probs), max(probs)) if len(probs) > 1 else (max(0.0, p - 0.15), min(1.0, p + 0.15))
            out.update(prob=round(100 * p), prob_lo=round(100 * lo), prob_hi=round(100 * hi), n_parts=len(probs),
                       level="много" if p >= 0.7 else ("заметно" if p >= 0.4 else "мало"))
    out["rows"].sort(key=lambda r: -abs(r.get("contribution") or 0))
    return out
