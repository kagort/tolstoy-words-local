"""Пополнение словаря: однокоренные слова из корпуса и автоматические категории.

Однокоренные. Формы слова («милосердны») находятся по лемме, если в словаре есть «милосердный»; поэтому предлагаются
однокоренные **леммы**: основа получается отсечением словообразовательного суффикса и окончания («милосердие» →
«милосерд»), учитываются приставки не-, без-/бес- и чередования на конце основы (страх → страш-ный). Кандидаты — только
леммы, которые встречаются в текстах; случайные совпадения начала («мил-», «сердце») отсекаются близостью векторов navec.

Категории. Слова без категории группируются k-средними по векторам navec; слова одной словообразовательной семьи
попадают в одну группу; название группы — главное существительное группы (ближайшее к центру), вариант — общее
существительное рядом с центром группы.
Слова без вектора относятся к группе по контекстам своих вхождений, иначе — «ПРОЧЕЕ». Пользователь правит и применяет.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import DictionaryEntry, Hit, TokenRow
from app.nlp.parsers import _morph
from app.nlp.tree import norm

# суффиксы и окончания, отсекаемые для основы (сначала длинные)
SUFFIXES = sorted({
    "ировать", "ованный", "ование", "овать", "евать", "ывать", "ивать", "ность", "ество", "ство", "тель", "ница",
    "ение", "ание", "ость", "есть", "ский", "ской", "енный", "анный", "ный", "ной", "ний", "ник", "ица", "ие", "ье", "ия",
    "ий", "ый", "ой", "ая", "ое", "ые", "ать", "ять", "ить", "еть", "уть", "ти", "ть", "ец", "ка", "о", "а", "я", "е", "ь",
    "й", "и", "ы", "у", "ю",
}, key=len, reverse=True)
PREFIXES = ("не", "без", "бес")
# чередования на конце основы: страх — страшный, друг — дружба, свет — свеча, простой — прощение
ALTERNATIONS = [("ск", "щ"), ("ст", "щ"), ("к", "ч"), ("г", "ж"), ("х", "ш"), ("т", "ч"), ("т", "щ"), ("д", "ж"),
                ("з", "ж"), ("с", "ш"), ("ц", "ч")]
MIN_STEM = 3
WORD = re.compile(r"^[а-яё-]+$")


def stems(lemma: str) -> set[str]:
    """Основы слова для поиска однокоренных: «милосердие» → {«милосерд»}, «страх» → {«страх», «страш»}."""
    w = norm(lemma.strip())
    if " " in w or not WORD.match(w):
        return set()
    if w.endswith(("ся", "сь")) and len(w) > 5:
        w = w[:-2]
    base = w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= MIN_STEM:
            base = w[: -len(suf)]
            break
    out = {base}
    for p in PREFIXES:
        if base.startswith(p) and len(base) - len(p) >= 4:
            out.add(base[len(p):])
    for b in sorted(out):  # копия: множество пополняется в цикле
        for a, c in ALTERNATIONS:
            if b.endswith(a):
                out.add(b[: -len(a)] + c)
            elif b.endswith(c) and len(b) - len(c) + len(a) >= MIN_STEM:
                out.add(b[: -len(c)] + a)
    return {b for b in out if len(b) >= MIN_STEM}


def _cos(a, b) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))


def cognates(session: Session, words: list[str], text_ids: list[int] | None, exclude: set[str] | None = None,
             per_word: int = 12) -> list[dict]:
    """Однокоренные леммы из текстов для каждого слова: [{"source", "items": [{lemma, upos, count, sim, example}]}].

    text_ids=None — все тексты. exclude — леммы, которые уже есть в словаре."""
    from app.services.semantic import vector

    exclude = {norm(x) for x in (exclude or set())}
    by_word = {w: stems(w) for w in dict.fromkeys(words) if stems(w)}
    patterns = sorted({p + s for ss in by_word.values() for s in ss for p in ("", *PREFIXES)})
    if not patterns:
        return []
    found: dict[str, tuple] = {}
    for k in range(0, len(patterns), 200):
        part = patterns[k : k + 200]
        q = (select(TokenRow.lemma_norm, func.mode().within_group(TokenRow.upos), func.count(), func.min(TokenRow.sentence_id))
             .where(or_(*[TokenRow.lemma_norm.like(p.replace("%", "") + "%") for p in part]),
                    TokenRow.upos.in_(["NOUN", "ADJ", "VERB", "ADV", "PROPN"])))
        if text_ids is not None:
            q = q.where(TokenRow.text_id.in_(text_ids or [-1]))
        for lemma, upos, c, sid in session.execute(q.group_by(TokenRow.lemma_norm)):
            found[lemma] = (upos, c, sid)
    out = []
    for w, ss in by_word.items():
        src = norm(w)
        vs = vector(src)
        items = []
        for lemma, (upos, c, sid) in found.items():
            if lemma == src or lemma in exclude or not WORD.match(lemma):
                continue
            stem = next((s for s in sorted(ss, key=len, reverse=True)
                         if any(lemma.startswith(p + s) for p in ("", *PREFIXES))), None)
            if stem is None or len(lemma) > len(stem) + 12:
                continue
            vc = vector(lemma)
            sim = _cos(vs, vc) if vs is not None and vc is not None else None
            # короткая основа совпадает случайно («рад-» — «радио»): нужна близость по смыслу
            if sim is not None and sim < (0.4 if len(stem) < 5 else 0.25):
                continue
            if sim is None and len(stem) < 5:
                continue
            items.append({"lemma": lemma, "upos": upos, "count": c, "sim": round(sim, 2) if sim is not None else None,
                          "example": sid})
        items.sort(key=lambda x: (-(x["sim"] if x["sim"] is not None else 0.2), -x["count"]))
        if items:
            out.append({"source": w, "items": items[:per_word]})
    return out


# ------------------------------------------------------------------ автоматические категории

OTHER = "ПРОЧЕЕ"


def _general_noun(centroid, members: set[str]) -> str | None:
    """Общее существительное рядом с центром группы: для «роза, сирень, ландыш» — «цветок»."""
    from app.services.semantic import nearest

    morph = _morph()
    for w, _ in nearest(centroid, 60):
        if len(w) < 3 or not WORD.match(w) or w in members:
            continue
        p = morph.parse(w)
        if p and p[0].normal_form == w and p[0].tag.POS == "NOUN" and "Name" not in str(p[0].tag) and "Surn" not in str(p[0].tag):
            return w
    return None


def suggest_categories(session: Session, dictionary_id: int, k: int | None = None, only_empty: bool = True) -> dict:
    """Группы слов словаря для категорий. only_empty — только записи без категории (существующие не трогаются)."""
    from app.services.semantic import kmeans, vector

    q = select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dictionary_id).order_by(DictionaryEntry.surface)
    entries = [e for e in session.scalars(q) if not (only_empty and e.category)]
    have, vecs, missing = [], [], []
    for e in entries:
        v = vector(e.lemma)
        if v is None:
            v = vector(e.surface)
        (have if v is not None else missing).append(e)
        if v is not None:
            vecs.append(v / (np.linalg.norm(v) or 1))
    if len(have) < 2:
        return {"groups": [], "k": 0, "n": len(entries), "missing": [e.surface for e in missing]}
    x = np.array(vecs)
    k = max(2, min(k or round(math.sqrt(len(have) / 2)), 20, len(have)))
    labels = kmeans(x, k).tolist()
    # одна словообразовательная семья — одна группа (по большинству)
    family = defaultdict(list)
    for i, e in enumerate(have):
        key = min(stems(e.lemma), key=len, default=None)
        if key:
            family[key].append(i)
    for idx in family.values():
        if len(idx) > 1:
            major = Counter(labels[i] for i in idx).most_common(1)[0][0]
            for i in idx:
                labels[i] = major
    centroids = {}
    for j in set(labels):
        m = x[[i for i in range(len(have)) if labels[i] == j]].mean(0)
        centroids[j] = m / (np.linalg.norm(m) or 1)
    # слова без вектора — по контекстам вхождений
    placed_missing: dict[int, int] = {}
    for e in missing:
        ctx = _context_vector(session, e.id)
        if ctx is not None:
            j, s = max(((j, float(ctx @ c)) for j, c in centroids.items()), key=lambda t: t[1])
            if s >= 0.3:
                placed_missing[e.id] = j
    groups = []
    for j in sorted(centroids, key=lambda j: -sum(1 for lab in labels if lab == j)):
        members = [have[i] for i in range(len(have)) if labels[i] == j]
        members.sort(key=lambda e: -float(x[have.index(e)] @ centroids[j]))
        extra = [e for e in missing if placed_missing.get(e.id) == j]
        surfaces = {norm(e.lemma) for e in members}
        # название — главное существительное группы (ближайшее к центру); вариант — общее слово рядом с центром
        head = next((e for e in members if e.upos == "NOUN" and " " not in e.lemma), members[0])
        general = _general_noun(centroids[j], surfaces)
        groups.append({"id": len(groups) + 1, "name": head.lemma.upper(),
                       "alt": [w.upper() for w in dict.fromkeys([general or "", members[0].lemma]) if w and w != head.lemma],
                       "entries": [{"id": e.id, "surface": e.surface, "context": False} for e in members]
                       + [{"id": e.id, "surface": e.surface, "context": True} for e in extra]})
    rest = [e for e in missing if e.id not in placed_missing]
    if rest:
        groups.append({"id": len(groups) + 1, "name": OTHER, "alt": [],
                       "entries": [{"id": e.id, "surface": e.surface, "context": False} for e in rest]})
    return {"groups": groups, "k": k, "n": len(entries), "missing": [e.surface for e in rest]}


def _context_vector(session: Session, entry_id: int, max_sentences: int = 200):
    """Средний вектор знаменательных слов предложений, где встречается слово (для слов, которых нет в navec)."""
    from app.services.semantic import vector

    sids = list(session.scalars(select(Hit.sentence_id).where(Hit.entry_id == entry_id).limit(max_sentences)))
    if not sids:
        return None
    lemmas = Counter(session.scalars(select(TokenRow.lemma_norm).where(
        TokenRow.sentence_id.in_(sids), TokenRow.upos.in_(["NOUN", "ADJ", "VERB", "ADV"]))))
    vs = [vector(w) * c for w, c in lemmas.most_common(300) if vector(w) is not None]
    if not vs:
        return None
    m = np.sum(vs, axis=0)
    return m / (np.linalg.norm(m) or 1)


def apply_categories(session: Session, dictionary_id: int, names: dict[int, str], assign: dict[int, int],
                     only_empty: bool = True) -> int:
    """names: номер группы → название категории; assign: запись → номер группы. Возвращает число записей с категорией."""
    n = 0
    for e in session.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dictionary_id,
                                                           DictionaryEntry.id.in_(list(assign) or [-1]))):
        if only_empty and e.category:
            continue
        name = (names.get(assign[e.id]) or "").strip()
        if name:
            e.category = name[:255]
            n += 1
    return n
