"""Словосочетания: «здравый смысл», «закон и порядок», «закон * порядок» (звёздочка — любое одно слово).

Одно описание используется в двух местах:
- поиск по словосочетанию в конкордансе (`concordance(..., phrase=...)`);
- словосочетания как записи словаря (`DictionaryEntry.phrase`): вхождение привязано к главному слову.

Слова сравниваются по лемме (все варианты разбора pymorphy3) или по форме, знаки препинания между словами
не считаются: «порядок, основанный на праве» находится по запросу «порядок основанный на право».
"""

from __future__ import annotations

import re

from sqlalchemy import and_, exists, or_, select

from app.models import TokenRow
from app.nlp.tree import norm

GAP = "*"
CONTENT = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")


def is_phrase(text: str | None) -> bool:
    return bool(text) and len(split_words(text)) > 1


def split_words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s,;:]+", (text or "").strip()) if w]


def parse(text: str) -> list[dict]:
    """Строка → слоты: {"lemmas": [...], "form": "...", "upos": [...]} или {"gap": 1} для «*»."""
    from app.nlp.morph import parse_options

    slots = []
    for w in split_words(text):
        if w == GAP:
            slots.append({"gap": 1})
            continue
        opts = [o for o in parse_options(w) if o.score >= 0.01] or parse_options(w)
        lemmas = sorted({norm(o.lemma) for o in opts} | {norm(w)})
        slots.append({"lemmas": lemmas, "form": norm(w), "upos": [o.upos for o in opts[:3]]})
    while slots and "gap" in slots[0]:
        slots.pop(0)
    while slots and "gap" in slots[-1]:
        slots.pop()
    return slots


def head_slot(slots: list[dict]) -> int:
    """Главное слово записи словаря: первое существительное, иначе первое знаменательное, иначе первое слово."""
    words = [k for k, s in enumerate(slots) if "gap" not in s]
    for pos_set in (("NOUN", "PROPN"), CONTENT):
        for k in words:
            if slots[k]["upos"] and slots[k]["upos"][0] in pos_set:
                return k
    return words[0] if words else 0


def _match_token(tok, slot: dict) -> bool:
    return norm(tok.lemma) in slot["lemmas"] or norm(tok.form) == slot["form"]


def find(tokens: list, slots: list[dict], any_order: bool = False) -> list[list]:
    """Все вхождения словосочетания в предложение. tokens — объекты с form, lemma, upos, idx (порядок предложения).

    Возвращает списки токенов-слов (без «*»). any_order — слова в любом порядке рядом друг с другом
    (окно = число слов словосочетания + 1).
    """
    words = [t for t in tokens if t.upos != "PUNCT"]
    if not slots or not words:
        return []
    if any_order:
        need = [s for s in slots if "gap" not in s]
        span = len(slots) + 1
        found = []
        for i in range(len(words)):
            window = words[i : i + span]
            used, picked = set(), []
            for s in need:
                tok = next((t for t in window if t.idx not in used and _match_token(t, s)), None)
                if tok is None:
                    break
                used.add(tok.idx)
                picked.append(tok)
            if len(picked) == len(need):
                found.append(sorted(picked, key=lambda t: t.idx))
        # одно и то же сочетание находится из нескольких окон — оставляем уникальные
        uniq = {tuple(t.idx for t in m): m for m in found}
        return list(uniq.values())
    found = []
    n = len(slots)
    for i in range(len(words)):
        if i + n > len(words):
            break
        picked = []
        for k, s in enumerate(slots):
            tok = words[i + k]
            if "gap" in s:
                continue
            if not _match_token(tok, s):
                picked = None
                break
            picked.append(tok)
        if picked:
            found.append(picked)
    return found


def contains_head(tokens: list, slots: list[dict], head: int, head_idx: int) -> bool:
    """Есть ли в предложении словосочетание, в котором главное слово — токен head_idx (для записей словаря)."""
    k = sum(1 for s in slots[:head] if "gap" not in s)  # номер главного слова среди найденных токенов
    return any(m[k].idx == head_idx for m in find(tokens, slots))


def sql_prefilter(slots: list[dict], sentence_col):
    """Быстрый отбор предложений, где есть все слова словосочетания (порядок проверяется потом в Python)."""
    from sqlalchemy.orm import aliased

    conds = []
    for s in slots:
        if "gap" in s:
            continue
        t = aliased(TokenRow)  # внешний запрос сам может содержать tokens — без псевдонима условие «слипнется» с ним
        conds.append(exists(select(t.id).where(
            t.sentence_id == sentence_col, or_(t.lemma_norm.in_(s["lemmas"]), t.form_norm == s["form"]))))
    return and_(*conds) if conds else None


def matched_text(sentence_text: str, match: list) -> str:
    """Отрывок предложения от первого до последнего слова словосочетания."""
    return sentence_text[match[0].start: match[-1].end]


def match_sentences(session, sentence_ids: list[int], slots: list[dict], any_order: bool = False,
                    positions: dict[int, set[int]] | None = None) -> dict[int, list[str]]:
    """Из предложений-кандидатов — те, где словосочетание действительно есть: {sentence_id: [найденные отрывки]}.

    positions (если передан словарь) заполняется номерами токенов словосочетания: {sentence_id: {idx, …}}.
    """
    from app.services.matching import load_sentences

    out: dict[int, list[str]] = {}
    ids = sorted(set(sentence_ids))
    for k in range(0, len(ids), 2000):
        for sid, sent in load_sentences(session, ids[k : k + 2000]).items():
            ms = find(sent.tokens, slots, any_order)
            if ms:
                out[sid] = [matched_text(sent.text, m) for m in ms]
                if positions is not None:
                    positions[sid] = {t.idx for m in ms for t in m}
    return out
