"""Статистический портрет текста: размер, лексика, частоты, морфология, синтаксис, пунктуация, динамика.

Результат кэшируется в texts.meta["stats"] и пересчитывается при повторном разборе текста
или при смене версии расчёта (STATS_VERSION).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from app.models import Dictionary, Hit, SentenceRow, TextDoc, TokenRow
from app.nlp.tree import norm, parse_feats
from app.services.stats import ipm

STATS_VERSION = "3"
POOL_SIZE = 600
TOP_LISTS = (("top_all", None, 30), ("top_noun", {"NOUN"}, 20), ("top_propn", {"PROPN"}, 15), ("top_verb", {"VERB"}, 20),
             ("top_adj", {"ADJ"}, 20), ("top_adv", {"ADV"}, 15))

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
NON_WORD = {"PUNCT", "SYM"}
VOWELS = set("аеёиоуыэюяaeiouy")
SEGMENTS = 20
MATTR_WINDOW = 500

PUNCT_NAMES = {
    ",": "запятая", ".": "точка", "!": "восклицательный знак", "?": "вопросительный знак", "…": "многоточие",
    "...": "многоточие", "—": "тире", "–": "тире", "-": "дефис/тире", ":": "двоеточие", ";": "точка с запятой",
    "«": "кавычки", "»": "кавычки", '"': "кавычки", "„": "кавычки", "“": "кавычки", "(": "скобки", ")": "скобки",
}

FEATURE_GROUPS = [
    # (заголовок, часть речи или None — для всех, признак)
    ("Падеж существительных", ("NOUN", "PROPN"), "Case"),
    ("Число существительных", ("NOUN", "PROPN"), "Number"),
    ("Время глаголов", ("VERB", "AUX"), "Tense"),
    ("Вид глаголов", ("VERB",), "Aspect"),
    ("Наклонение", ("VERB", "AUX"), "Mood"),
    ("Лицо глаголов (перспектива повествования)", ("VERB", "AUX"), "Person"),
    ("Залог", ("VERB",), "Voice"),
    ("Форма глагола", ("VERB",), "VerbForm"),
    ("Степень сравнения прилагательных", ("ADJ",), "Degree"),
    ("Полные и краткие прилагательные", ("ADJ",), "Variant"),
]

FEATURE_VALUE_LABELS = {
    "Nom": "именительный", "Gen": "родительный", "Dat": "дательный", "Acc": "винительный", "Ins": "творительный",
    "Loc": "предложный", "Par": "партитив", "Voc": "звательный",
    "Sing": "ед. число", "Plur": "мн. число",
    "Past": "прошедшее", "Pres": "настоящее", "Fut": "будущее",
    "Perf": "совершенный", "Ind": "изъявительное", "Cnd": "условное",
    ("Aspect", "Imp"): "несовершенный", ("Mood", "Imp"): "повелительное",
    "1": "1-е лицо", "2": "2-е лицо", "3": "3-е лицо",
    "Act": "действительный", "Pass": "страдательный", "Mid": "средний (-ся)",
    "Fin": "личная форма", "Inf": "инфинитив", "Part": "причастие", "Conv": "деепричастие",
    "Pos": "положительная", "Cmp": "сравнительная", "Sup": "превосходная", "Short": "краткая форма", "Full": "полная форма",
}


def syllables(word: str) -> int:
    return sum(1 for ch in word.lower() if ch in VOWELS)


def mattr(seq: list[str], window: int = MATTR_WINDOW) -> float | None:
    """Moving-Average TTR (Covington & McFall): средний TTR в скользящем окне — не зависит от длины текста."""
    n = len(seq)
    if n == 0:
        return None
    if n <= window:
        return len(set(seq)) / n
    counts = Counter(seq[:window])
    total = len(counts) / window
    for i in range(window, n):
        out, inc = seq[i - window], seq[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        total += len(counts) / window
    return total / (n - window + 1)


def yule_k(freqs: list[int]) -> float:
    """K Юла: чем больше, тем больше повторов (беднее словарь)."""
    n = sum(freqs)
    if n == 0:
        return 0.0
    spectrum = Counter(freqs)
    s2 = sum(m * m * vm for m, vm in spectrum.items())
    return 10_000 * (s2 - n) / (n * n)


def zipf_slope(freqs_sorted: list[int], max_rank: int = 1000) -> float | None:
    """Наклон прямой log(частота) ~ log(ранг) — показатель закона Ципфа (около −1 для естественных текстов)."""
    pts = [(math.log(r), math.log(f)) for r, f in enumerate(freqs_sorted[:max_rank], start=1) if f > 0]
    if len(pts) < 10:
        return None
    mx = statistics.mean(x for x, _ in pts)
    my = statistics.mean(y for _, y in pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return round(num / den, 3) if den else None


def compute(session: Session, text_id: int) -> dict:
    doc = session.get(TextDoc, text_id)
    T = TokenRow

    # ------------------------------------------------ последовательность слов
    rows = session.execute(
        select(T.form, T.lemma_norm, T.upos).where(T.text_id == text_id, T.upos.not_in(NON_WORD))
        .order_by(T.sentence_id, T.idx)
    ).all()
    words = [r.form for r in rows]
    lemmas = [r.lemma_norm for r in rows]
    n_words = len(words)
    n_tokens = session.scalar(select(func.count()).select_from(T).where(T.text_id == text_id)) or 0
    n_sent = doc.n_sentences or 0
    n_par = (session.scalar(select(func.max(SentenceRow.paragraph_idx)).where(SentenceRow.text_id == text_id)) or -1) + 1
    word_lens = [len(w) for w in words]
    syl = [syllables(w) for w in words]

    # ------------------------------------------------ лексика и частоты
    lemma_freq = Counter(lemmas)
    lemma_pos = {}
    for r in rows:
        lemma_pos.setdefault(r.lemma_norm, Counter())[r.upos] += 1
    freqs_sorted = sorted(lemma_freq.values(), reverse=True)
    n_types_forms = len({w.lower().replace("ё", "е") for w in words})
    n_lemmas = len(lemma_freq)
    hapax = sum(1 for f in lemma_freq.values() if f == 1)
    dis = sum(1 for f in lemma_freq.values() if f == 2)
    content = sum(1 for r in rows if r.upos in CONTENT_POS)

    def top(pos_set: set[str] | None, k: int = 25):
        items = []
        for lemma, f in lemma_freq.most_common():
            main_pos = lemma_pos[lemma].most_common(1)[0][0]
            if pos_set is None or main_pos in pos_set:
                items.append({"lemma": lemma, "count": f, "ipm": round(ipm(f, n_words), 1), "pos": main_pos})
            if len(items) >= k:
                break
        return items

    spectrum = Counter(lemma_freq.values())
    lexical = {
        "n_forms": n_types_forms,
        "n_lemmas": n_lemmas,
        "ttr_forms": round(n_types_forms / n_words, 4) if n_words else 0,
        "ttr_lemmas": round(n_lemmas / n_words, 4) if n_words else 0,
        "mattr": round(mattr(lemmas) or 0, 4),
        "hapax": hapax,
        "hapax_pct": round(100 * hapax / n_lemmas, 1) if n_lemmas else 0,
        "dis": dis,
        "yule_k": round(yule_k(list(lemma_freq.values())), 2),
        "herdan_c": round(math.log(n_lemmas) / math.log(n_words), 4) if n_words > 1 and n_lemmas > 1 else None,
        "guiraud": round(n_lemmas / math.sqrt(n_words), 2) if n_words else None,
        "density": round(100 * content / n_words, 1) if n_words else 0,
        "zipf_slope": zipf_slope(freqs_sorted),
    }
    frequency = {
        "top_all": top(None, 30),
        "top_noun": top({"NOUN"}, 20),
        "top_propn": top({"PROPN"}, 15),
        "top_verb": top({"VERB"}, 20),
        "top_adj": top({"ADJ"}, 20),
        "top_adv": top({"ADV"}, 15),
        # запас частотных лемм — из него списки пересобираются при фильтре стоп-слов (frequency_lists)
        "pool": top(None, POOL_SIZE),
        "zipf": [[r, f] for r, f in enumerate(freqs_sorted[:2000], start=1)],
        "spectrum": [[m, spectrum[m]] for m in sorted(spectrum)[:15]],
    }

    # ------------------------------------------------ морфология
    pos_counts = Counter(r.upos for r in rows)
    feat_rows = session.execute(
        select(T.upos, T.feats, func.count()).where(T.text_id == text_id, T.upos.not_in(NON_WORD)).group_by(T.upos, T.feats)
    ).all()
    groups = []
    for title, pos_set, feat in FEATURE_GROUPS:
        c = Counter()
        for upos, feats, cnt in feat_rows:
            if upos in pos_set:
                v = parse_feats(feats).get(feat)
                if v:
                    c[v] += cnt
        if feat == "Variant":  # у полных прилагательных признака нет: полные = все ADJ − краткие
            full = sum(cnt for upos, _, cnt in feat_rows if upos in pos_set) - c.get("Short", 0)
            if full > 0:
                c["Full"] = full
        total = sum(c.values())
        if total:
            groups.append({"title": title, "feature": feat, "total": total,
                           "values": [{"value": v, "label": FEATURE_VALUE_LABELS.get((feat, v), FEATURE_VALUE_LABELS.get(v, v)), "count": n,
                                       "pct": round(100 * n / total, 1)} for v, n in c.most_common()]})
    nouns = pos_counts["NOUN"] + pos_counts["PROPN"]
    verbs = pos_counts["VERB"] + pos_counts["AUX"]
    morphology = {
        "pos": [{"upos": p, "count": n, "pct": round(100 * n / n_words, 2)} for p, n in pos_counts.most_common()],
        "noun_verb": round(nouns / verbs, 2) if verbs else None,
        "adj_noun": round(pos_counts["ADJ"] / nouns, 2) if nouns else None,
        "pron_share": round(100 * pos_counts["PRON"] / n_words, 2) if n_words else 0,
        "features": groups,
    }

    # ------------------------------------------------ синтаксис
    S = SentenceRow
    sent_rows = session.execute(
        select(S.idx, S.n_words, S.tree_depth, S.n_clauses, S.sent_type, S.is_direct_speech, S.has_homogeneous,
               S.has_participial, S.has_gerund).where(S.text_id == text_id).order_by(S.idx)
    ).all()
    lens = [r.n_words for r in sent_rows]
    mdd = session.scalar(
        select(func.avg(func.abs(T.head - T.idx))).where(T.text_id == text_id, T.head > 0, T.upos.not_in(NON_WORD))
    )
    deprels = session.execute(
        select(T.deprel, func.count()).where(T.text_id == text_id, T.upos.not_in(NON_WORD)).group_by(T.deprel).order_by(func.count().desc())
    ).all()
    n_s = len(sent_rows) or 1
    types = Counter(r.sent_type for r in sent_rows)
    hist = Counter(min(n // 5 * 5, 100) for n in lens)
    syntax = {
        "avg_len": round(statistics.mean(lens), 2) if lens else 0,
        "median_len": statistics.median(lens) if lens else 0,
        "stdev_len": round(statistics.pstdev(lens), 2) if lens else 0,
        "max_len": max(lens) if lens else 0,
        "avg_depth": round(statistics.mean(r.tree_depth for r in sent_rows), 2) if sent_rows else 0,
        "avg_clauses": round(statistics.mean(r.n_clauses for r in sent_rows), 2) if sent_rows else 0,
        "mdd": round(float(mdd), 3) if mdd is not None else None,
        "types": [{"type": t, "pct": round(100 * c / n_s, 1), "count": c} for t, c in types.most_common()],
        "direct_speech": round(100 * sum(r.is_direct_speech for r in sent_rows) / n_s, 1),
        "homogeneous": round(100 * sum(r.has_homogeneous for r in sent_rows) / n_s, 1),
        "participial": round(100 * sum(r.has_participial for r in sent_rows) / n_s, 1),
        "gerund": round(100 * sum(r.has_gerund for r in sent_rows) / n_s, 1),
        "hist": [[b, round(100 * hist[b] / n_s, 2)] for b in sorted(hist)],
        "deprels": [{"deprel": d, "count": c, "pct": round(100 * c / n_words, 2)} for d, c in deprels[:25]],
    }

    # ------------------------------------------------ пунктуация (на 1000 слов)
    punct = session.execute(
        select(T.form, func.count()).where(T.text_id == text_id, T.upos == "PUNCT").group_by(T.form)
    ).all()
    pc = Counter()
    for form, cnt in punct:
        pc[PUNCT_NAMES.get(form, form)] += cnt
    punctuation = [{"mark": m, "count": c, "per1000": round(1000 * c / n_words, 1) if n_words else 0} for m, c in pc.most_common(15)]

    # ------------------------------------------------ удобочитаемость
    asl = n_words / n_sent if n_sent else 0
    asw = statistics.mean(syl) if syl else 0
    readability = {
        "asl": round(asl, 2),
        "asw": round(asw, 3),
        # формула Флеша, адаптированная для русского (И. В. Оборнева): чем выше, тем легче текст
        "flesch_oborneva": round(206.835 - 1.3 * asl - 60.1 * asw, 1) if n_words else None,
    }

    # ------------------------------------------------ динамика по ходу текста (20 равных частей)
    seg = defaultdict(lambda: {"len": [], "ds": 0, "n": 0, "depth": []})
    for r in sent_rows:
        b = min(SEGMENTS - 1, (r.idx - 1) * SEGMENTS // max(n_sent, 1))
        s = seg[b]
        s["len"].append(r.n_words)
        s["depth"].append(r.tree_depth)
        s["ds"] += r.is_direct_speech
        s["n"] += 1
    seg_words = {b: sum(v["len"]) for b, v in seg.items()}
    dynamics = {
        "segments": [f"{5 * b + 5}%" for b in range(SEGMENTS)],
        "avg_len": [round(statistics.mean(seg[b]["len"]), 2) if seg[b]["n"] else None for b in range(SEGMENTS)],
        "avg_depth": [round(statistics.mean(seg[b]["depth"]), 2) if seg[b]["n"] else None for b in range(SEGMENTS)],
        "direct_speech": [round(100 * seg[b]["ds"] / seg[b]["n"], 1) if seg[b]["n"] else None for b in range(SEGMENTS)],
        "seg_words": [seg_words.get(b, 0) for b in range(SEGMENTS)],
    }

    return {
        "version": STATS_VERSION,
        "pipeline": doc.pipeline_version,
        "size": {
            "tokens": n_tokens, "words": n_words, "sentences": n_sent, "paragraphs": n_par,
            "chars": len(doc.norm_text or ""), "avg_word_len": round(statistics.mean(word_lens), 2) if word_lens else 0,
            "avg_syllables": round(asw, 2), "long_words_pct": round(100 * sum(1 for s in syl if s >= 4) / n_words, 1) if n_words else 0,
            "word_len_hist": sorted(Counter(min(x, 15) for x in word_lens).items()),
        },
        "lexical": lexical,
        "frequency": frequency,
        "morphology": morphology,
        "syntax": syntax,
        "punctuation": punctuation,
        "readability": readability,
        "dynamics": dynamics,
    }


def dictionary_dynamics(session: Session, text_id: int, n_sent: int, seg_words: list[int]) -> list[dict]:
    """Плотность слов каждого словаря по ходу текста (ipm в каждой из 20 частей). Не кэшируется: словари меняются."""
    S = SentenceRow
    bucket = func.least(SEGMENTS - 1, func.floor((S.idx - 1) * SEGMENTS / float(max(n_sent, 1)))).cast(Integer)
    out = []
    for d in session.scalars(select(Dictionary).order_by(Dictionary.id)):
        cnt = dict(session.execute(
            select(bucket, func.count()).select_from(Hit).join(S, S.id == Hit.sentence_id)
            .where(Hit.text_id == text_id, Hit.dictionary_id == d.id,
                   (Hit.is_valid.is_(None)) | (Hit.is_valid.is_(True)))
            .group_by(bucket)
        ).all())
        if cnt:
            out.append({"name": d.name, "ipm": [round(ipm(cnt.get(b, 0), seg_words[b]), 1) for b in range(SEGMENTS)]})
    return out


def get_stats(session: Session, text_id: int, refresh: bool = False) -> dict | None:
    doc = session.get(TextDoc, text_id)
    if doc is None or doc.status != "ready":
        return None
    stats = (doc.meta or {}).get("stats")
    if refresh or not stats or stats.get("version") != STATS_VERSION or stats.get("pipeline") != doc.pipeline_version:
        stats = compute(session, text_id)
        doc.meta = {**(doc.meta or {}), "stats": stats}
        session.commit()
    stats = dict(stats)
    stats["dynamics"] = {**stats["dynamics"],
                         "dictionaries": dictionary_dynamics(session, text_id, stats["size"]["sentences"], stats["dynamics"]["seg_words"])}
    return stats


def frequency_lists(stats: dict, stop: frozenset[str] = frozenset()) -> dict:
    """Частотные списки портрета без стоп-слов (пересобираются из запаса `pool`, кэш не пересчитывается)."""
    fq = dict(stats.get("frequency") or {})
    pool = fq.get("pool")
    if not stop or not pool:
        return fq
    items = [x for x in pool if norm(x["lemma"]) not in stop]
    for key, pos_set, k in TOP_LISTS:
        fq[key] = [x for x in items if pos_set is None or x["pos"] in pos_set][:k]
    return fq
