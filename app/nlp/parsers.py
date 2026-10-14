"""Морфологический и синтаксический разбор.

Два бэкенда с одинаковым интерфейсом:
- ``stanza``  — Stanza, модель SynTagRus (точнее, рекомендуется для работы);
- ``natasha`` — Natasha/Slovnet (лёгкий и быстрый, модели входят в pip-пакет).

Вход — уже токенизированные предложения (список списков строк),
выход — список списков ``Parsed`` с леммой, UPOS, граммемами, головой и типом связи.
Дерево после разбора «чинится» (sanitize_tree): ровно один корень, без циклов.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class Parsed:
    form: str
    lemma: str
    upos: str
    feats: dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = "root"


class Parser(Protocol):
    name: str
    version: str

    def parse(self, sentences: list[list[str]]) -> list[list[Parsed]]: ...


# ---------------------------------------------------------------- лемматизация (для Natasha)

_UD2PYMORPHY = {
    "NOUN": {"NOUN"},
    "PROPN": {"NOUN"},
    "ADJ": {"ADJF", "ADJS", "COMP"},
    "VERB": {"VERB", "INFN", "PRTF", "PRTS", "GRND"},
    "AUX": {"VERB", "INFN"},
    "ADV": {"ADVB", "COMP", "PRED"},
    "NUM": {"NUMR"},
    "PRON": {"NPRO"},
    "DET": {"ADJF", "NPRO"},
    "ADP": {"PREP"},
    "CCONJ": {"CONJ"},
    "SCONJ": {"CONJ"},
    "PART": {"PRCL"},
    "INTJ": {"INTJ"},
}

_CASES = {"Nom": "nomn", "Gen": "gent", "Dat": "datv", "Acc": "accs", "Ins": "ablt", "Loc": "loct"}
_NUMBERS = {"Sing": "sing", "Plur": "plur"}
_GENDERS = {"Masc": "masc", "Fem": "femn", "Neut": "neut"}


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemmatize_cached(form: str, upos: str, feats_key: tuple) -> str:
    if upos in ("PUNCT", "SYM", "X") or not any(ch.isalpha() for ch in form):
        return form
    feats = dict(feats_key)
    allowed = _UD2PYMORPHY.get(upos)
    best, best_score = None, -1.0
    for p in _morph().parse(form):
        score = p.score
        if allowed and p.tag.POS in allowed:
            score += 1.0
        grams = p.tag.grammemes
        for ud, table in (("Case", _CASES), ("Number", _NUMBERS), ("Gender", _GENDERS)):
            v = feats.get(ud)
            if v and table.get(v) in grams:
                score += 0.2
        if score > best_score:
            best, best_score = p, score
    if best is None:
        return form.lower()
    if upos == "PROPN":
        return best.normal_form.capitalize()
    return best.normal_form


def lemmatize(form: str, upos: str, feats: dict[str, str]) -> str:
    return _lemmatize_cached(form, upos, tuple(sorted(feats.items())))


_FIXABLE_UPOS = {"NOUN", "VERB", "AUX", "ADJ", "ADV", "NUM", "PRON", "DET"}
_CYRILLIC = re.compile(r"^[а-яё-]+$", re.IGNORECASE)


def fix_lemma(form: str, lemma: str, upos: str, feats: dict[str, str]) -> str:
    """Исправляет несуществующие леммы нейросетевого лемматизатора (Stanza: «шёл» → «ийти»).

    Если лемма неизвестна словарю pymorphy3, а pymorphy3 даёт для словоформы известную
    лемму той же части речи — берём её. Редкие слова, которых нет в словаре, не трогаем.
    """
    if upos not in _FIXABLE_UPOS or not lemma or not _CYRILLIC.match(lemma):
        return lemma
    morph = _morph()
    if morph.word_is_known(lemma.lower()):
        return lemma
    alt = lemmatize(form, upos, feats)
    if alt and alt.lower() != lemma.lower() and morph.word_is_known(alt.lower()):
        return alt
    return lemma


# ---------------------------------------------------------------- починка дерева


def sanitize_tree(words: list[Parsed]) -> list[Parsed]:
    """Гарантирует корректное дерево: один корень, нет петель и циклов."""
    n = len(words)
    if n == 0:
        return words
    for i, w in enumerate(words, start=1):
        if w.head == i or w.head < 0 or w.head > n:
            w.head = 0
    roots = [i for i, w in enumerate(words, start=1) if w.head == 0]

    def pick_root(cands: list[int]) -> int:
        for pref in ("VERB", "AUX", "ADJ", "NOUN", "PROPN"):
            for c in cands:
                if words[c - 1].upos == pref:
                    return c
        for c in cands:
            if words[c - 1].upos != "PUNCT":
                return c
        return cands[0]

    # разрыв циклов: если из узла нельзя дойти до 0 — подвешиваем к временному корню
    def reaches_root(i: int) -> bool:
        seen = set()
        while i != 0:
            if i in seen:
                return False
            seen.add(i)
            i = words[i - 1].head
        return True

    root = pick_root(roots) if roots else pick_root(list(range(1, n + 1)))
    words[root - 1].head = 0
    words[root - 1].deprel = "root"
    for i in range(1, n + 1):
        if i == root:
            continue
        w = words[i - 1]
        if w.head == 0:
            w.head = root
            if w.deprel == "root":
                w.deprel = "parataxis" if w.upos != "PUNCT" else "punct"
        elif not reaches_root(i):
            w.head = root
            if w.deprel == "root":
                w.deprel = "parataxis"
    return words


# ---------------------------------------------------------------- Natasha


class NatashaParser:
    name = "natasha"

    def __init__(self) -> None:
        from natasha import NewsEmbedding, NewsMorphTagger, NewsSyntaxParser
        import natasha

        emb = NewsEmbedding()
        self._morph = NewsMorphTagger(emb)
        self._syntax = NewsSyntaxParser(emb)
        self.version = f"natasha-{getattr(natasha, '__version__', '1.6')}"

    def parse(self, sentences: list[list[str]]) -> list[list[Parsed]]:
        out: list[list[Parsed]] = []
        if not sentences:
            return out
        for mk, sk in zip(self._morph.map(sentences), self._syntax.map(sentences)):
            words = []
            for mt, st in zip(mk.tokens, sk.tokens):
                feats = dict(mt.feats or {})
                head = int(st.head_id) if str(st.head_id).isdigit() else 0
                words.append(
                    Parsed(
                        form=mt.text,
                        lemma=lemmatize(mt.text, mt.pos, feats),
                        upos=mt.pos,
                        feats=feats,
                        head=head,
                        deprel=st.rel or "dep",
                    )
                )
            out.append(sanitize_tree(words))
        return out


# ---------------------------------------------------------------- Stanza


class StanzaParser:
    name = "stanza"

    def __init__(self, model_dir: str | None = None, use_gpu: bool = False) -> None:
        import stanza

        kwargs = dict(
            lang="ru",
            processors="tokenize,pos,lemma,depparse",
            tokenize_pretokenized=True,
            use_gpu=use_gpu,
            logging_level="WARN",
        )
        if model_dir:
            kwargs["dir"] = model_dir
            kwargs["download_method"] = None  # модели уже скачаны при сборке образа — без обращения к сети
        self._nlp = stanza.Pipeline(**kwargs)
        self.version = f"stanza-{stanza.__version__}-ru"

    def parse(self, sentences: list[list[str]]) -> list[list[Parsed]]:
        if not sentences:
            return []
        doc = self._nlp(sentences)
        out = []
        for s in doc.sentences:
            words = []
            for w in s.words:
                feats = {}
                if w.feats:
                    for part in w.feats.split("|"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            feats[k] = v
                words.append(
                    Parsed(
                        form=w.text,
                        lemma=fix_lemma(w.text, w.lemma or w.text, w.upos or "X", feats),
                        upos=w.upos or "X",
                        feats=feats,
                        head=int(w.head or 0),
                        deprel=w.deprel or "dep",
                    )
                )
            out.append(sanitize_tree(words))
        return out


_PARSER: Parser | None = None


def get_parser(backend: str | None = None) -> Parser:
    """Ленивая загрузка парсера (модели загружаются один раз на процесс)."""
    global _PARSER
    from app.config import settings

    backend = (backend or settings.nlp_backend).lower()
    if _PARSER is not None and _PARSER.name == backend:
        return _PARSER
    if backend == "stanza":
        _PARSER = StanzaParser(model_dir=settings.stanza_dir or None)
    elif backend == "natasha":
        _PARSER = NatashaParser()
    else:
        raise ValueError(f"Неизвестный NLP_BACKEND: {backend}")
    log.info("NLP-парсер загружен: %s", _PARSER.version)
    return _PARSER
