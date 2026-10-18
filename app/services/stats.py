"""Статистические меры корпусной лингвистики (чистые функции)."""

from __future__ import annotations

import math


def ipm(count: int, corpus_words: int) -> float:
    """Частота на миллион словоупотреблений."""
    return count * 1_000_000 / corpus_words if corpus_words else 0.0


def log_likelihood(a: int, b: int, n1: int, n2: int) -> float:
    """G² (Dunning) для частоты a в корпусе размера n1 против b в n2."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    e1 = n1 * (a + b) / (n1 + n2)
    e2 = n2 * (a + b) / (n1 + n2)
    g2 = 0.0
    if a > 0 and e1 > 0:
        g2 += a * math.log(a / e1)
    if b > 0 and e2 > 0:
        g2 += b * math.log(b / e2)
    return 2 * g2


def log_ratio(a: int, b: int, n1: int, n2: int, smoothing: float = 0.5) -> float:
    """Двоичный логарифм отношения относительных частот (Hardie), со сглаживанием."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return math.log2(((a + smoothing) / n1) / ((b + smoothing) / n2))


def log_dice(fxy: int, fx: int, fy: int) -> float:
    """logDice (Rychlý): 14 + log2(2·fxy / (fx + fy)); максимум 14."""
    if fxy <= 0 or fx + fy <= 0:
        return 0.0
    return 14 + math.log2(2 * fxy / (fx + fy))


def gries_dp(counts: list[int], sizes: list[int]) -> float:
    """DP Грайса: 0 — слово равномерно распределено по частям корпуса, ближе к 1 — сосредоточено в немногих."""
    total = sum(counts)
    total_size = sum(sizes)
    if total == 0 or total_size == 0:
        return 0.0
    return 0.5 * sum(abs(c / total - s / total_size) for c, s in zip(counts, sizes))


SIGNIFICANCE_LL = 3.84  # p < 0.05 при одной степени свободы


# ------------------------------------------------------------------ меры ассоциации (коллокации)

MEASURES = {
    "count": "частота сочетания",
    "logdice": "logDice",
    "pmi": "PMI",
    "npmi": "NPMI",
    "mi3": "MI³",
    "tscore": "t-score",
    "g2": "G² (log-likelihood)",
    "dp_fwd": "ΔP слово → коллокат",
    "dp_bwd": "ΔP коллокат → слово",
}
MEASURE_HELP = {
    "count": "сколько раз коллокат встретился рядом со словом (в окне или в синтаксической связи)",
    "logdice": "сила связи до 14; не зависит от размера корпуса — удобна для сравнения корпусов",
    "pmi": "насколько чаще случайного слова встречаются вместе; завышает редкие сочетания",
    "npmi": "PMI, нормированная в диапазон −1…1 (1 — встречаются только вместе)",
    "mi3": "PMI с поправкой на частоту — меньше доверяет редким сочетаниям",
    "tscore": "уверенность, что сочетание не случайно; выше у частых сочетаний",
    "g2": "статистическая значимость (≥ 3,84 — p < 0,05; ≥ 10,83 — p < 0,001); отрицательная — реже ожидаемого",
    "dp_fwd": "направленная мера: насколько слово «предсказывает» коллокат (−1…1)",
    "dp_bwd": "направленная мера: насколько коллокат «предсказывает» слово (−1…1)",
}


def association(c: int, fx: int, fy: int, n: int, span: int = 1) -> dict:
    """Меры ассоциации по таблице сопряжённости 2×2 (Evert 2005).

    c — частота сочетания; fx — частота слова (узла); fy — частота коллоката в корпусе; n — размер корпуса;
    span — число позиций окна вокруг узла (для оконных коллокаций: 2·окно; для синтаксических связей — 1).
    Строки таблицы: позиции рядом с узлом (R1 = fx·span) / остальные; столбцы: коллокат / другие слова.
    """
    r1 = max(fx * span, c, 1)
    c1 = max(fy, c, 1)
    n = max(n, r1 + c1)
    o11, o12, o21 = c, r1 - c, c1 - c
    o22 = max(n - r1 - c1 + c, 0)
    e11 = r1 * c1 / n
    cells = ((o11, e11), (o12, r1 * (n - c1) / n), (o21, (n - r1) * c1 / n), (o22, (n - r1) * (n - c1) / n))
    g2 = 2 * sum(o * math.log(o / e) for o, e in cells if o > 0 and e > 0)
    if o11 < e11:
        g2 = -g2
    out = {"count": c, "expected": round(e11, 3)}
    if c > 0:
        pmi = math.log2(o11 / e11)
        p_xy = o11 / n
        out.update(
            pmi=round(pmi, 2),
            npmi=round(pmi / -math.log2(p_xy), 3) if p_xy < 1 else 1.0,
            mi3=round(math.log2(o11 ** 3 / e11), 2),
            tscore=round((o11 - e11) / math.sqrt(o11), 2),
            logdice=round(log_dice(c, fx, fy), 2),
        )
    else:
        out.update(pmi=None, npmi=None, mi3=None, tscore=None, logdice=None)
    out["g2"] = round(g2, 2)
    out["dp_fwd"] = round(o11 / r1 - o21 / (n - r1), 4) if n > r1 else 0.0
    out["dp_bwd"] = round(o11 / c1 - o12 / (n - c1), 4) if n > c1 else 0.0
    return out


# ------------------------------------------------------------------ дисперсия по частям текста


def juilland_d(counts: list[int]) -> float | None:
    """D Жуйяна по n равным частям: 1 — равномерно, 0 — всё в одной части."""
    n = len(counts)
    if n < 2:
        return None
    mean = sum(counts) / n
    if mean == 0:
        return None
    sd = math.sqrt(sum((x - mean) ** 2 for x in counts) / n)
    return max(0.0, 1 - (sd / mean) / math.sqrt(n - 1))


def tf_idf(tf: int, doc_len: int, df: int, n_docs: int) -> float:
    """TF-IDF: относительная частота в тексте × сглаженный логарифм обратной документной частоты."""
    if not doc_len or not n_docs:
        return 0.0
    return (tf / doc_len) * (math.log((1 + n_docs) / (1 + df)) + 1)
