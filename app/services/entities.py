"""Имена собственные (NER): извлечение моделью Natasha/Slovnet (входит в пакет natasha, ~2 МБ),
приведение к начальной форме, указатель, сеть персонажей, связь мест со словарём.

Классы: PER — персоны, LOC — места (география), ORG — организации и прочие названия.
Ручные правки (EntityOverride) применяются при показе: сменить класс, слить варианты имени, скрыть ошибку.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from functools import lru_cache

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Entity, EntityOverride, Hit, SentenceRow, TextDoc, TokenRow
from app.nlp.parsers import _morph
from app.nlp.tree import parse_feats
from app.services import analytics as A
from app.services.jobs import set_progress

log = logging.getLogger(__name__)

ETYPES = {"PER": "персоны", "LOC": "места", "ORG": "организации и названия"}
ETYPE_ONE = {"PER": "персона", "LOC": "место", "ORG": "организация / название"}
UNITS = {"sentence": "в одном предложении", "paragraph": "в одном абзаце"}
BATCH = 400

_CASE = {"Nom": "nomn", "Gen": "gent", "Dat": "datv", "Acc": "accs", "Ins": "ablt", "Loc": "loct"}
_NAME_TAGS = {"Name", "Surn", "Patr", "Geox", "Orgn"}


@lru_cache(maxsize=1)
def embedding():
    """Векторы navec (общие для NER и семантики; загружаются один раз на процесс, ~1,5 с)."""
    from natasha import NewsEmbedding

    return NewsEmbedding()


@lru_cache(maxsize=1)
def ner_tagger():
    from natasha import NewsNERTagger

    return NewsNERTagger(embedding())


def _cap(src: str, word: str) -> str:
    if src.isupper() and len(src) > 1:
        return word.upper()
    return word[:1].upper() + word[1:] if src[:1].isupper() else word


def _nominative(form: str, upos: str, feats: dict, etype: str, gender: str | None) -> str:
    """Начальная форма слова в составе имени: падеж — именительный, род и число сохраняются."""
    if upos == "PUNCT" or not any(ch.isalpha() for ch in form):
        return form
    parses = _morph().parse(form)
    case = _CASE.get(feats.get("Case", ""))

    def score(p):
        s = p.score
        if case and case in p.tag:
            s += 1
        if _NAME_TAGS & set(p.tag.grammemes):
            s += 0.5
        if upos == "ADJ" and p.tag.POS in ("ADJF", "PRTF"):
            s += 0.5
        return s

    best = max(parses, key=score) if parses else None
    if best is None:
        return form
    want = {"nomn"}
    if etype == "PER":
        want.add("sing")
    elif feats.get("Number") == "Plur":
        want.add("plur")
    if upos == "ADJ" and gender and "plur" not in want:
        want.add({"Masc": "masc", "Fem": "femn", "Neut": "neut"}.get(gender, "masc"))
    infl = best.inflect(want) or best.inflect({"nomn"})
    return _cap(form, infl.word if infl else form)


def normalize(tokens: list[tuple[str, str, dict]], etype: str) -> str:
    """tokens — [(форма, UPOS, граммемы)]; слова в кавычках не изменяются («Правда»)."""
    out, quoted = [], False
    noun_gender = next((f.get("Gender") for _, u, f in tokens if u in ("NOUN", "PROPN") and f.get("Gender")), None)
    for form, upos, feats in tokens:
        if form in ("«", "„", '"', "\u201c"):
            quoted = True
            out.append(form)
            continue
        if form in ("»", "\u201d"):
            quoted = False
            out.append(form)
            continue
        out.append(form if quoted else _nominative(form, upos, feats, etype, noun_gender))
    text = " ".join(out)
    for a, b in ((" »", "»"), ("« ", "«"), (" - ", "-"), (" .", "."), (" ,", ",")):
        text = text.replace(a, b)
    return text.strip()


# ------------------------------------------------------------------ извлечение


def run_ner(job_id: int | None, text_id: int) -> int:
    """Задача «ner»: имена собственные одного текста (старые упоминания текста заменяются)."""
    tagger = ner_tagger()
    with SessionLocal() as s:
        s.execute(delete(Entity).where(Entity.text_id == text_id))
        s.commit()
        sids = list(s.scalars(select(SentenceRow.id).where(SentenceRow.text_id == text_id).order_by(SentenceRow.idx)))
    total = 0
    from app.services.ingest import _copy

    for k in range(0, len(sids), BATCH):
        chunk = sids[k : k + BATCH]
        with SessionLocal() as s:
            texts = dict(s.execute(select(SentenceRow.id, SentenceRow.text).where(SentenceRow.id.in_(chunk))).all())
            toks = defaultdict(list)
            for sid, form, upos, feats, st, en in s.execute(
                select(TokenRow.sentence_id, TokenRow.form, TokenRow.upos, TokenRow.feats, TokenRow.start_char, TokenRow.end_char)
                .where(TokenRow.sentence_id.in_(chunk)).order_by(TokenRow.sentence_id, TokenRow.idx)
            ):
                toks[sid].append((form, upos, parse_feats(feats), st, en))
            order = [sid for sid in chunk if texts.get(sid)]
            rows = []
            for sid, markup in zip(order, tagger.map([texts[sid] for sid in order])):
                for sp in markup.spans:
                    span_toks = [(f, u, ft) for f, u, ft, st, en in toks.get(sid, []) if st >= sp.start and en <= sp.stop]
                    surface = texts[sid][sp.start : sp.stop]
                    norm_name = normalize(span_toks, sp.type) if span_toks else surface
                    if not any(ch.isalpha() for ch in norm_name):
                        continue
                    rows.append((text_id, sid, sp.type, sp.start, sp.stop, surface, norm_name))
            _copy(s, "entities", ("text_id", "sentence_id", "etype", "start_char", "end_char", "surface", "norm"), rows)
            s.commit()
            total += len(rows)
        if job_id:
            set_progress(job_id, min(1.0, (k + len(chunk)) / max(len(sids), 1)),
                         f"Имена собственные: {min(k + len(chunk), len(sids))} из {len(sids)} предложений")
    with SessionLocal() as s:  # отметка «имена извлечены» — и для текстов, где имён нет вовсе (слияние JSONB в SQL — без гонок)
        from sqlalchemy import text as sql

        s.execute(sql("UPDATE texts SET meta = coalesce(meta, '{}'::jsonb) || jsonb_build_object('ner', jsonb_build_object('count', :n)) "
                      "WHERE id = :id"), {"n": total, "id": text_id})
        s.commit()
    log.info("Текст %s: имён собственных %s", text_id, total)
    return total


def texts_without_entities(session: Session, text_ids: list[int]) -> list[int]:
    done = set(session.scalars(select(Entity.text_id).where(Entity.text_id.in_(text_ids or [-1])).distinct()))
    done |= {tid for tid, meta in session.execute(select(TextDoc.id, TextDoc.meta).where(TextDoc.id.in_(text_ids or [-1])))
             if (meta or {}).get("ner")}
    return [t for t in text_ids if t not in done]


# ------------------------------------------------------------------ правки и указатель


def overrides(session: Session) -> dict[str, EntityOverride]:
    return {o.norm: o for o in session.scalars(select(EntityOverride))}


def resolve(norm_name: str, etype: str, ovr: dict) -> tuple[str, str, bool]:
    """(каноническое имя, класс, скрыто) с учётом правок; слияния прослеживаются по цепочке."""
    seen = set()
    name, typ, hidden = norm_name, etype, False
    while name in ovr and name not in seen:
        seen.add(name)
        o = ovr[name]
        hidden = hidden or o.hidden
        typ = o.etype or typ
        if o.canonical and o.canonical != name:
            name = o.canonical
        else:
            break
    return name, typ, hidden


def entity_index(session: Session, text_ids: list[int], etype: str | None = None, query: str = "", top: int = 300) -> dict:
    ovr = overrides(session)
    rows = session.execute(
        select(Entity.norm, Entity.etype, func.count(), func.count(func.distinct(Entity.text_id)), func.min(Entity.sentence_id))
        .where(Entity.text_id.in_(text_ids or [-1])).group_by(Entity.norm, Entity.etype)
    ).all()
    agg: dict[str, dict] = {}
    by_type = Counter()
    for norm_name, typ, c, n_texts, first in rows:
        name, typ2, hidden = resolve(norm_name, typ, ovr)
        if hidden:
            continue
        a = agg.setdefault(name, {"name": name, "types": Counter(), "count": 0, "texts": 0, "variants": Counter(), "first": first})
        a["types"][typ2] += c
        a["count"] += c
        a["texts"] = max(a["texts"], n_texts)
        a["variants"][norm_name] += c
        a["first"] = min(a["first"], first)
    items = []
    for a in agg.values():
        typ = a["types"].most_common(1)[0][0]
        by_type[typ] += 1
        if etype and typ != etype:
            continue
        if query and query.lower() not in a["name"].lower():
            continue
        items.append({"name": a["name"], "etype": typ, "label": ETYPE_ONE.get(typ, typ), "count": a["count"], "texts": a["texts"],
                      "variants": [v for v, _ in a["variants"].most_common(4) if v != a["name"]], "first": a["first"],
                      "edited": a["name"] in ovr or any(v in ovr for v in a["variants"])})
    items.sort(key=lambda x: -x["count"])
    return {"items": items[:top], "total": len(items), "by_type": dict(by_type),
            "n_mentions": sum(x["count"] for x in items), "missing": texts_without_entities(session, text_ids)}


def _canonical_rows(session: Session, text_ids: list[int], etype: str | None = None):
    """Упоминания с каноническими именами: (sentence_id, text_id, paragraph, idx, name, type, start, end)."""
    ovr = overrides(session)
    q = (select(Entity.sentence_id, Entity.text_id, SentenceRow.paragraph_idx, SentenceRow.idx, Entity.norm, Entity.etype,
                Entity.start_char, Entity.end_char)
         .join(SentenceRow, SentenceRow.id == Entity.sentence_id).where(Entity.text_id.in_(text_ids or [-1])))
    for sid, tid, par, idx, norm_name, typ, st, en in session.execute(q):
        name, typ2, hidden = resolve(norm_name, typ, ovr)
        if hidden or (etype and typ2 != etype):
            continue
        yield sid, tid, par, idx, name, typ2, st, en


def mentions(session: Session, text_ids: list[int], name: str, limit: int = 100) -> dict:
    found = [r for r in _canonical_rows(session, text_ids) if r[4] == name]
    sids = sorted({r[0] for r in found})[:limit]
    marks = defaultdict(list)
    for r in found:
        if r[0] in sids:
            marks[r[0]].append((r[6], r[7]))
    return {"total": len(found), "sentences": _sentences(session, sids, marks)}


def _sentences(session: Session, sids: list[int], marks: dict) -> list[dict]:
    rows = session.execute(select(SentenceRow.id, SentenceRow.text, SentenceRow.idx, TextDoc.title, TextDoc.author)
                           .join(TextDoc, TextDoc.id == SentenceRow.text_id).where(SentenceRow.id.in_(sids or [-1]))
                           .order_by(SentenceRow.text_id, SentenceRow.idx)).all()
    return [{"id": r.id, "text": r.text, "idx": r.idx, "source": f"{r.author + ', ' if r.author else ''}«{r.title}»",
             "marks": sorted(marks.get(r.id, []))} for r in rows]


def set_override(session: Session, norm_name: str, etype: str | None = None, canonical: str | None = None,
                 hidden: bool = False) -> None:
    o = session.get(EntityOverride, norm_name) or EntityOverride(norm=norm_name)
    o.etype = etype if etype in ETYPES else None
    o.canonical = (canonical or "").strip() or None
    o.hidden = hidden
    if not (o.etype or o.canonical or o.hidden):
        if session.get(EntityOverride, norm_name):
            session.delete(o)
        return
    session.add(o)


# ------------------------------------------------------------------ сеть персонажей и места × словарь


def _unit(tid, par, idx, unit):
    return (tid, par) if unit == "paragraph" else (tid, idx)


def persons_network(session: Session, text_ids: list[int], unit: str = "sentence", top: int = 30, min_weight: int = 2) -> dict:
    members = defaultdict(set)
    counts = Counter()
    for sid, tid, par, idx, name, typ, st, en in _canonical_rows(session, text_ids, "PER"):
        members[_unit(tid, par, idx, unit)].add(name)
        counts[name] += 1
    names = [n for n, _ in counts.most_common(top)]
    keep = set(names)
    edges = Counter()
    for group in members.values():
        g = sorted(group & keep)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                edges[(g[i], g[j])] += 1
    edge_list = [{"a": a, "b": b, "w": w} for (a, b), w in edges.most_common() if w >= min_weight]
    pos = spring_layout(names, [(e["a"], e["b"], e["w"]) for e in edge_list])
    degree = Counter()
    for e in edge_list:
        degree[e["a"]] += e["w"]
        degree[e["b"]] += e["w"]
    nodes = [{"name": n, "count": counts[n], "degree": degree[n], "x": pos[n][0], "y": pos[n][1]} for n in names]
    return {"nodes": nodes, "edges": edge_list, "unit": unit}


def spring_layout(names: list[str], edges: list[tuple[str, str, int]], iters: int = 300, seed: int = 7) -> dict:
    """Силовая укладка графа (Fruchterman–Reingold) на numpy, детерминированная."""
    import numpy as np

    n = len(names)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    idx = {name: i for i, name in enumerate(names)}
    w = np.zeros((n, n))
    for a, b, wt in edges:
        w[idx[a], idx[b]] = w[idx[b], idx[a]] = math.log1p(wt)
    if w.max() > 0:
        w = w / w.max()
    k = 1.5 / math.sqrt(n)
    t = 0.3
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
        rep = (k * k / dist ** 2)[:, :, None] * delta
        att = (w * dist / k)[:, :, None] * delta
        # притяжение к центру: несвязанные группы (герои разных книг) не разлетаются
        disp = rep.sum(1) - att.sum(1) - 0.5 * pos * np.linalg.norm(pos, axis=1, keepdims=True)
        length = np.maximum(np.linalg.norm(disp, axis=1), 1e-6)
        pos += disp / length[:, None] * np.minimum(length, t)[:, None]
        pos -= pos.mean(0)
        t *= 0.985
    scale = np.abs(pos).max() or 1
    return {name: (round(float(pos[i, 0] / scale), 4), round(float(pos[i, 1] / scale), 4)) for name, i in idx.items()}


def entity_dictionary(session: Session, text_ids: list[int], dictionary_id: int, entry_ids: list[int] | None = None,
                      etype: str = "LOC", unit: str = "sentence", top: int = 20, top_words: int = 12) -> dict:
    """Матрица «имя (место / персона) × слово словаря»: вхождения слова в том же предложении или абзаце."""
    from app.models import DictionaryEntry

    units = defaultdict(set)
    counts = Counter()
    for sid, tid, par, idx, name, typ, st, en in _canonical_rows(session, text_ids, etype):
        units[_unit(tid, par, idx, unit)].add(name)
        counts[name] += 1
    hq = (select(Hit.entry_id, Hit.text_id, SentenceRow.paragraph_idx, SentenceRow.idx)
          .join(SentenceRow, SentenceRow.id == Hit.sentence_id)
          .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(text_ids or [-1]), A.VALID))
    if entry_ids:
        hq = hq.where(Hit.entry_id.in_(entry_ids))
    cell = Counter()
    word_tot = Counter()
    for eid, tid, par, idx in session.execute(hq):
        for name in units.get(_unit(tid, par, idx, unit), ()):
            cell[(name, eid)] += 1
            word_tot[eid] += 1
    names = sorted({n for n, _ in cell}, key=lambda n: -sum(c for (m, _), c in cell.items() if m == n))[:top]
    eids = [e for e, _ in word_tot.most_common(top_words)] if not entry_ids else [e for e in entry_ids if word_tot.get(e)]
    surf = dict(session.execute(select(DictionaryEntry.id, DictionaryEntry.surface).where(DictionaryEntry.id.in_(eids or [-1]))).all())
    return {"names": names, "words": [{"entry_id": e, "word": surf.get(e, str(e))} for e in eids],
            "z": [[cell.get((n, e), 0) for e in eids] for n in names], "mentions": {n: counts[n] for n in names},
            "etype": etype, "unit": unit}


def cooccurrence_examples(session: Session, text_ids: list[int], name: str, entry_id: int,
                          unit: str = "sentence", limit: int = 60) -> list[dict]:
    """Предложения, где имя и слово словаря встречаются в одном предложении (или абзаце); оба подсвечены."""
    ent = [r for r in _canonical_rows(session, text_ids) if r[4] == name]
    hits = session.execute(select(Hit.sentence_id, Hit.text_id, SentenceRow.paragraph_idx, SentenceRow.idx, TokenRow.start_char, TokenRow.end_char)
                           .join(SentenceRow, SentenceRow.id == Hit.sentence_id).join(TokenRow, TokenRow.id == Hit.token_id)
                           .where(Hit.entry_id == entry_id, Hit.text_id.in_(text_ids or [-1]), A.VALID)).all()
    both = {_unit(r[1], r[2], r[3], unit) for r in ent} & {_unit(t, p, i, unit) for _, t, p, i, _, _ in hits}
    marks = defaultdict(list)
    for sid, tid, par, idx, st, en in hits:
        if _unit(tid, par, idx, unit) in both:
            marks[sid].append((st, en))
    for r in ent:
        if _unit(r[1], r[2], r[3], unit) in both:
            marks[r[0]].append((r[6], r[7]))
    return _sentences(session, sorted(marks)[:limit], marks)
