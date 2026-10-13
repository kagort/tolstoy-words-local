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
