"""Верификация словаря через pymorphy3: варианты разбора слова и соответствие UD."""

from __future__ import annotations

from dataclasses import dataclass

from app.nlp.parsers import _morph
from app.nlp.tree import norm

# pymorphy3 POS → UD UPOS (для фильтра при сопоставлении)
PYMORPHY2UD = {
    "NOUN": "NOUN",
    "ADJF": "ADJ",
    "ADJS": "ADJ",
    "COMP": "ADJ",
    "VERB": "VERB",
    "INFN": "VERB",
    "PRTF": "VERB",
    "PRTS": "VERB",
    "GRND": "VERB",
    "NUMR": "NUM",
    "ADVB": "ADV",
    "NPRO": "PRON",
    "PRED": "ADV",
    "PREP": "ADP",
    "CONJ": "CCONJ",
    "PRCL": "PART",
    "INTJ": "INTJ",
}

POS_LABELS = {
    "NOUN": "существительное",
    "PROPN": "имя собственное",
    "ADJ": "прилагательное",
    "VERB": "глагол",
    "AUX": "вспомогательный глагол",
    "ADV": "наречие",
    "NUM": "числительное",
    "PRON": "местоимение",
    "DET": "местоимение-прилагательное",
    "ADP": "предлог",
    "CCONJ": "сочинительный союз",
    "SCONJ": "подчинительный союз",
    "PART": "частица",
    "INTJ": "междометие",
    "PUNCT": "знак препинания",
    "SYM": "символ",
    "X": "другое",
}


def pos_label(upos: str | None) -> str:
    """«существительное (NOUN)»."""
    if not upos:
        return "любая часть речи"
    name = POS_LABELS.get(upos)
    return f"{name} ({upos})" if name else upos


@dataclass
class ParseOption:
    lemma: str
    upos: str
    tag: str
    score: float

    @property
    def key(self) -> str:
        return f"{self.lemma}|{self.upos}"


def parse_options(word: str) -> list[ParseOption]:
    """Уникальные варианты (лемма, UPOS) с суммарной вероятностью."""
    agg: dict[tuple[str, str], ParseOption] = {}
    for p in _morph().parse(word.strip()):
        upos = PYMORPHY2UD.get(p.tag.POS or "", "X")
        if "Name" in p.tag or "Surn" in p.tag or "Geox" in p.tag:
            upos = "PROPN"
        key = (p.normal_form, upos)
        if key in agg:
            agg[key].score += p.score
        else:
            agg[key] = ParseOption(p.normal_form, upos, str(p.tag), p.score)
    return sorted(agg.values(), key=lambda o: -o.score)


def word_forms(lemma: str, upos: str | None) -> set[str]:
    """Все формы слова (нормализованные) для режима сопоставления «по формам»."""
    forms: set[str] = set()
    for p in _morph().parse(lemma):
        if p.normal_form != lemma:
            continue
        if upos and PYMORPHY2UD.get(p.tag.POS or "", "X") != upos and not (upos == "PROPN" and p.tag.POS == "NOUN"):
            continue
        for f in p.lexeme:
            forms.add(norm(f.word))
    return forms or {norm(lemma)}
