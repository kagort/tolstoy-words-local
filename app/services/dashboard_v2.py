"""Дашборд v2 «корпус × словарь»: девять разделов с выводом одной фразой над каждым графиком (docs/PROGRESS.md,
волна «Визуализация»; оформление — docs/DESIGN_TOKENS.md). Классический интерфейс показывает прежний дашборд
(services/dashboard.py); общие части (плитки, ключевые слова, персонажи, места) берутся оттуда.

Разделы: 1 покрытие словаря; 2 ядро и периферия (частота × равномерность); 3 профиль категорий по группам
(радар ≤ 6 групп, иначе тепловая карта отклонений); 4 характерные слова групп (log-ratio + G²); 5 время;
6 синтаксические роли по категориям; 7 сеть совместной встречаемости слов словаря; 8 персонажи и места × категории;
9 качество данных; плюс цитаты — примеры употребления частых слов.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Dictionary, DictionaryEntry, Hit, SentenceRow, TextDoc
from app.services import analytics as A
from app.services.stats import SIGNIFICANCE_LL, log_likelihood, log_ratio

INITIAL_SPACE = re.compile(r"(\b[A-ZА-ЯЁ]\.) (?=[A-ZА-ЯЁ])")  # пробел после инициала перед следующим инициалом или фамилией
MAX_CAT_COLORS = 8  # категорий в цвете; остальные — серым (DESIGN_TOKENS.md, раздел 2)
ROLE_ORDER = ["subject", "predicate", "object", "attribute", "adverbial", "other"]
ROLE_NAMES = {"subject": "подлежащее", "predicate": "сказуемое", "object": "дополнение", "attribute": "определение",
              "adverbial": "обстоятельство", "other": "прочее"}


def _fmt(n: float, digits: int = 0) -> str:
    """Число по-русски: неразрывный пробел в тысячах, запятая в дробях."""
    s = f"{n:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def build_v2(session: Session, sel, dictionary_id: int | None) -> dict:
    from app.services.dashboard import build

    base = build(session, sel, dictionary_id)  # плитки, ключевые слова, персонажи, места, тексты — как в прежнем дашборде
    d = session.get(Dictionary, dictionary_id) if dictionary_id else None
    ids = sel.ids
    base["v2"] = None
    if d is None or not ids:
        return base
    from app.services.dictionaries import no_categories

    flat = no_categories(d)
    freq = A.frequency(session, d.id, ids)
    rows = freq["rows"]
    n_words = freq["corpus_words"] or 1
    found = [r for r in rows if r["count"]]
    cat_of = {r["entry_id"]: ("" if flat else (r["category"] or "")) for r in rows}
    cat_totals = Counter()
    for r in found:
        cat_totals[cat_of[r["entry_id"]] or A.NO_CATEGORY] += r["count"]
    has_cats = not flat and any(c != A.NO_CATEGORY for c in cat_totals)
    cat_order = [c for c, _ in cat_totals.most_common()] if has_cats else []
    cat_color = {c: (i if i < MAX_CAT_COLORS and c != A.NO_CATEGORY else None) for i, c in enumerate(cat_order)}
    texts = list(session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from, TextDoc.n_words)
                                 .where(TextDoc.id.in_(ids))))
    v2 = {"has_cats": has_cats, "categories": [{"name": c, "color": cat_color[c], "count": cat_totals[c]} for c in cat_order]}
    v2["coverage"] = _coverage(found, rows, n_words)
    v2["core"] = _core(found, cat_of, cat_color, len(ids))
    groups = _groups(texts)
    v2["group_kind"] = groups["kind"]
    v2["profile"] = _profile(session, d.id, ids, groups, cat_order) if has_cats else None
    v2["distinct"] = _distinct_words(session, d.id, ids, groups, rows, n_words)
    v2["time"] = _time(session, d.id, ids, texts, has_cats, cat_order, found)
    v2["syntax"] = _syntax(session, d.id, ids, cat_of, has_cats, cat_order, found)
    v2["network"] = _network(session, d.id, ids, rows, cat_of, cat_color)
    v2["names"] = _names(session, d.id, ids, found, cat_of, has_cats, cat_order, cat_color)
    v2["quality"] = _quality(session, d, ids, rows)
    v2["quotes"] = _quotes(session, d.id, ids, found[:12])
    for sec in v2.values():  # «Л. Н. Толстой» во фразах-выводах не разрывается переносом строки
        if isinstance(sec, dict) and sec.get("say"):
            sec["say"] = INITIAL_SPACE.sub("\\1\u00a0", sec["say"])
    base["v2"] = v2
    return base


# ------------------------------------------------------------------ 1. покрытие


def _coverage(found: list[dict], rows: list[dict], n_words: int) -> dict:
    hits = sum(r["count"] for r in found)
    dead = [r["word"] for r in rows if not r["count"]]
    rare = [r for r in found if r["count"] <= 2]
    pct = 100 * hits / n_words
    say = (f"Найдено {len(found)} из {len(rows)} слов словаря; они составляют {_fmt(pct, 2)} % слов выборки"
           f" ({_fmt(hits)} {_plural(hits, 'вхождение', 'вхождения', 'вхождений')}).")
    if dead:
        say += f" {len(dead)} {_plural(len(dead), 'слово', 'слова', 'слов')} не встретилось ни разу."
    return {"found": len(found), "entries": len(rows), "hits": hits, "pct": round(pct, 3), "dead": dead[:40],
            "n_dead": len(dead), "n_rare": len(rare), "say": say}


# ------------------------------------------------------------------ 2. ядро и периферия


def _core(found: list[dict], cat_of: dict, cat_color: dict, n_texts: int) -> dict | None:
    pts = [r for r in found if r["dp"] is not None]
    if n_texts < 3 or len(pts) < 3:
        return None
    med = statistics.median(r["ipm"] for r in pts)
    core = sorted([r for r in pts if r["ipm"] >= med and r["dp"] <= 0.5], key=lambda r: -r["ipm"])
    periph = [r for r in pts if r["dp"] >= 0.8]
    labeled = {r["entry_id"] for r in sorted(pts, key=lambda r: -r["ipm"])[:8]}
    say = (f"Ядро словаря — частые и равномерно распределённые слова: {', '.join('«' + r['word'] + '»' for r in core[:5])}."
           if core else "Частых и равномерно распределённых слов мало: словарь держится на отдельных текстах.")
    if periph:
        say += f" {len(periph)} {_plural(len(periph), 'слово сосредоточено', 'слова сосредоточены', 'слов сосредоточены')} в немногих текстах (DP ≥ 0,8)."
    return {"say": say, "points": [{"word": r["word"], "ipm": r["ipm"], "dp": r["dp"], "count": r["count"], "texts": r["texts"],
                                    "cat": cat_of.get(r["entry_id"]) or "", "color": cat_color.get(cat_of.get(r["entry_id"]) or ""),
                                    "label": r["entry_id"] in labeled} for r in pts]}


# ------------------------------------------------------------------ группы выборки


def _groups(texts) -> dict:
    """Группы для сравнения: авторы (если их ≥ 2), иначе тексты (если их 2–40)."""
    authors = Counter(t.author for t in texts if t.author)
    if len(authors) >= 2:
        from app.services.labels import short_author

        members = defaultdict(list)
        for t in texts:
            members[short_author(t.author) if t.author else "без автора"].append(t.id)
        return {"kind": "author", "label": "авторы", "members": dict(members)}
    if 2 <= len(texts) <= 40:
        from app.services.labels import short_title

        return {"kind": "text", "label": "тексты", "members": {short_title(t.title): [t.id] for t in texts}}
    return {"kind": None, "label": "", "members": {}}


def _group_words(session: Session, members: dict) -> dict:
    ids = [t for v in members.values() for t in v]
    sizes = dict(session.execute(select(TextDoc.id, TextDoc.n_words).where(TextDoc.id.in_(ids or [-1]))).all())
    return {g: sum(sizes.get(t) or 0 for t in v) for g, v in members.items()}


def _group_counts(session: Session, dictionary_id: int, members: dict):
    """Вхождения «группа × слово словаря»."""
    group_of = {t: g for g, v in members.items() for t in v}
    out: dict[str, Counter] = defaultdict(Counter)
    for eid, tid, n in session.execute(select(Hit.entry_id, Hit.text_id, func.count())
                                       .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(list(group_of) or [-1]), A.valid())
                                       .group_by(Hit.entry_id, Hit.text_id)):
        out[group_of[tid]][eid] += n
    return out


# ------------------------------------------------------------------ 3. профиль категорий


def _profile(session: Session, dictionary_id: int, ids: list[int], groups: dict, cat_order: list[str]) -> dict | None:
    members = groups["members"]
    if not members:
        return None
    counts = _group_counts(session, dictionary_id, members)
    cat_of = dict(session.execute(select(DictionaryEntry.id, DictionaryEntry.category)
                                  .where(DictionaryEntry.dictionary_id == dictionary_id)).all())
    cats = [c for c in cat_order if c != A.NO_CATEGORY][:10]
    sizes = _group_words(session, members)
    names = sorted(members, key=lambda g: -sum(counts[g].values()))[:40]
    share, ipm_m = {}, {}
    for g in names:
        by_cat = Counter()
        for eid, n in counts[g].items():
            by_cat[cat_of.get(eid) or A.NO_CATEGORY] += n
        tot = sum(by_cat.values()) or 1
        share[g] = [round(100 * by_cat[c] / tot, 1) for c in cats]
        ipm_m[g] = [round(A.ipm(by_cat[c], sizes.get(g) or 1), 1) for c in cats]
    avg = [round(statistics.mean(share[g][k] for g in names), 1) for k in range(len(cats))] if names else []
    # z-оценки ipm по каждой категории — для тепловой карты отклонений
    z = {g: [] for g in names}
    for k in range(len(cats)):
        col = [ipm_m[g][k] for g in names]
        mu, sd = statistics.mean(col), (statistics.pstdev(col) or 1)
        for g in names:
            z[g].append(round((ipm_m[g][k] - mu) / sd, 2))
    notes = []
    for g in names[:3]:
        diff = [(share[g][k] - avg[k], cats[k]) for k in range(len(cats))]
        dv, c = max(diff)
        if dv > 0.5:
            notes.append(f"у {g} выше среднего «{c}» (+{_fmt(dv, 1)} п. п.)")
    say = ("Доли категорий по группам: " + "; ".join(notes) + ".") if notes else "Профили групп близки к среднему."
    form = "radar" if 2 <= len(names) <= 6 else ("heatmap" if len(names) > 6 else "bars")
    return {"say": say, "form": form, "cats": cats, "groups": names, "share": share, "avg": avg, "z": z, "ipm": ipm_m}


# ------------------------------------------------------------------ 4. характерные слова групп


def _distinct_words(session: Session, dictionary_id: int, ids: list[int], groups: dict, rows: list[dict], n_words: int) -> dict | None:
    members = groups["members"]
    if len(members) < 2:
        return None
    counts = _group_counts(session, dictionary_id, members)
    sizes = _group_words(session, members)
    word = {r["entry_id"]: r["word"] for r in rows}
    total = Counter()
    for g in counts:
        total.update(counts[g])
    names = sorted(members, key=lambda g: -sizes.get(g, 0))[:12]
    out = {}
    for g in names:
        na, nb = sizes.get(g) or 1, max(1, n_words - (sizes.get(g) or 0))
        items = []
        for eid, tot in total.items():
            if tot < 3:
                continue
            a = counts[g].get(eid, 0)
            b = tot - a
            items.append({"word": word.get(eid, str(eid)), "lr": round(log_ratio(a, b, na, nb), 2),
                          "ll": round(log_likelihood(a, b, na, nb), 2), "a": a, "b": b})
        items = [x for x in items if x["ll"] >= SIGNIFICANCE_LL / 2]
        pos = sorted([x for x in items if x["lr"] > 0], key=lambda x: -x["lr"])[:8]
        neg = sorted([x for x in items if x["lr"] < 0], key=lambda x: x["lr"])[:8]
        out[g] = {"items": sorted(pos + neg, key=lambda x: -x["lr"])}
    best = max(names, key=lambda g: sum(1 for x in out[g]["items"] if x["lr"] > 0 and x["ll"] >= SIGNIFICANCE_LL), default=None)
    pos = [x["word"] for x in out[best]["items"] if x["lr"] > 0 and x["ll"] >= SIGNIFICANCE_LL][:3] if best else []
    say = (f"Сильнее всех от остальных отличается {best}: чаще {', '.join('«' + w + '»' for w in pos)}." if pos
           else "Значимых различий в словаре между группами мало.")
    pair = None
    if len(names) == 2:  # две группы — рассеяние в стиле Scattertext
        a, b = names
        pair = {"a": a, "b": b, "points": [{"word": word.get(eid, str(eid)),
                                            "x": round(A.ipm(counts[a].get(eid, 0) + 0.5, sizes.get(a) or 1), 2),
                                            "y": round(A.ipm(counts[b].get(eid, 0) + 0.5, sizes.get(b) or 1), 2),
                                            "n": tot} for eid, tot in total.items() if tot >= 2]}
    return {"say": say, "groups": names, "by_group": out, "pair": pair, "sig": SIGNIFICANCE_LL}


# ------------------------------------------------------------------ 5. время


def _time(session: Session, dictionary_id: int, ids: list[int], texts, has_cats: bool, cat_order: list[str], found) -> dict | None:
    decades = sorted({(t.year_from // 10) * 10 for t in texts if t.year_from})
    if len(decades) < 3:
        return None
    by = "cat" if has_cats else "word"
    entry_ids = [] if has_cats else [r["entry_id"] for r in found[:8]]
    ov = A.overview(session, dictionary_id, ids, entry_ids, dim="period", step=10, by=by, groups_top=None)
    series = [{"name": w["word"], "ipm": w["ipm"]} for w in ov["words"]][:10]
    labels = [g["label"] for g in ov["groups"]]
    trends = []
    for s in series:
        vals = [v for v in s["ipm"] if v is not None]
        if len(vals) >= 3 and vals[0] + vals[-1] > 0:
            ch = (vals[-1] - vals[0]) / max(vals[0], 1)
            trends.append((ch, s["name"]))
    up = [n for c, n in sorted(trends, reverse=True) if c > 0.3][:2]
    down = [n for c, n in sorted(trends) if c < -0.3][:2]
    parts = []
    if up:
        parts.append("растут " + ", ".join("«" + n + "»" for n in up))
    if down:
        parts.append("убывают " + ", ".join("«" + n + "»" for n in down))
    say = (f"С {labels[0]} по {labels[-1]}: " + "; ".join(parts) + ".") if parts else "Заметных изменений во времени нет."
    return {"say": say, "periods": labels, "series": series, "n_texts": [g["n_texts"] for g in ov["groups"]]}


# ------------------------------------------------------------------ 6. синтаксис


def _syntax(session: Session, dictionary_id: int, ids: list[int], cat_of: dict, has_cats: bool, cat_order: list[str], found) -> dict | None:
    q = (select(Hit.entry_id, Hit.role, func.count()).where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(ids), A.valid())
         .group_by(Hit.entry_id, Hit.role))
    word = {r["entry_id"]: r["word"] for r in found}
    keys = [c for c in cat_order if c != A.NO_CATEGORY][:8] if has_cats else [r["word"] for r in found[:10]]
    if not keys:
        return None
    m: dict[str, Counter] = defaultdict(Counter)
    for eid, role, n in session.execute(q):
        k = (cat_of.get(eid) or A.NO_CATEGORY) if has_cats else word.get(eid)
        if k in keys:
            m[k][role if role in ROLE_ORDER else "other"] += n
    rows = []
    for k in keys:
        tot = sum(m[k].values())
        if tot:
            rows.append({"name": k, "total": tot, "pct": [round(100 * m[k][r] / tot, 1) for r in ROLE_ORDER]})
    if not rows:
        return None
    subj = max(rows, key=lambda r: r["pct"][0])
    attr = max(rows, key=lambda r: r["pct"][3])
    say = (f"Чаще всего подлежащим бывает «{subj['name']}» ({_fmt(subj['pct'][0], 0)} %), определением — "
           f"«{attr['name']}» ({_fmt(attr['pct'][3], 0)} %).")
    return {"say": say, "roles": [ROLE_NAMES[r] for r in ROLE_ORDER], "rows": rows, "by": "категориям" if has_cats else "словам"}


# ------------------------------------------------------------------ 7. сеть совместной встречаемости


def _network(session: Session, dictionary_id: int, ids: list[int], rows: list[dict], cat_of: dict, cat_color: dict,
             top: int = 40, max_rows: int = 300_000) -> dict | None:
    from app.services.entities import spring_layout

    by_sent: dict[int, set] = defaultdict(set)
    for sid, eid in session.execute(select(Hit.sentence_id, Hit.entry_id)
                                    .where(Hit.dictionary_id == dictionary_id, Hit.text_id.in_(ids), A.valid()).limit(max_rows)):
        by_sent[sid].add(eid)
    freq = Counter()
    for s in by_sent.values():
        freq.update(s)
    nodes = [e for e, _ in freq.most_common(top)]
    keep = set(nodes)
    pairs = Counter()
    for s in by_sent.values():
        s = sorted(s & keep)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pairs[(s[i], s[j])] += 1
    edges = [(a, b, w) for (a, b), w in pairs.most_common(90) if w >= 2]
    if len(edges) < 2:
        return None
    word = {r["entry_id"]: r["word"] for r in rows}
    used = sorted({a for a, _, _ in edges} | {b for _, b, _ in edges})
    pos = spring_layout([str(e) for e in used], [(str(a), str(b), w) for a, b, w in edges])
    a, b, w = edges[0]
    say = (f"Слова словаря чаще всего встречаются вместе в парах «{word.get(a)}» — «{word.get(b)}» ({w} "
           f"{_plural(w, 'предложение', 'предложения', 'предложений')}); всего {len(edges)} устойчивых связей.")
    return {"say": say,
            "nodes": [{"id": e, "word": word.get(e, str(e)), "x": pos[str(e)][0], "y": pos[str(e)][1], "n": freq[e],
                       "cat": cat_of.get(e) or "", "color": cat_color.get(cat_of.get(e) or "")} for e in used],
            "edges": [{"a": a, "b": b, "w": w} for a, b, w in edges]}


# ------------------------------------------------------------------ 8. персонажи и места × словарь


def _names(session: Session, dictionary_id: int, ids: list[int], found, cat_of: dict, has_cats: bool, cat_order: list[str],
           cat_color: dict) -> dict | None:
    from app.services.entities import entity_dictionary

    eids = [r["entry_id"] for r in found]
    if not eids:
        return None
    keys = [c for c in cat_order if c != A.NO_CATEGORY][:10] if has_cats else [r["word"] for r in found[:12]]
    word = {r["entry_id"]: r["word"] for r in found}

    def agg(ed):
        mats = []
        for n, row in zip(ed["names"], ed["z"]):
            acc = Counter()
            for w, v in zip(ed["words"], row):
                k = (cat_of.get(w["entry_id"]) or A.NO_CATEGORY) if has_cats else word.get(w["entry_id"])
                acc[k] += v
            mats.append([acc.get(k, 0) for k in keys])
        return mats

    per = entity_dictionary(session, ids, dictionary_id, eids, "PER", "sentence", top=14, top_words=len(eids))
    loc = entity_dictionary(session, ids, dictionary_id, eids, "LOC", "sentence", top=40, top_words=len(eids))
    if not per["names"] and not loc["names"]:
        return None
    per_rows = [(n, row) for n, row in zip(per["names"], agg(per)) if sum(row)]  # без пустых строк в тепловой карте
    per_names, per_z = [n for n, _ in per_rows], [row for _, row in per_rows]
    out = {"keys": keys, "persons": per_names, "z": per_z, "say": ""}
    if per_names:
        best = max(((v, n, keys[k]) for n, row in zip(per_names, per_z) for k, v in enumerate(row)), default=(0, "", ""))
        if best[0]:
            what = "словами категории" if has_cats else "словом"
            out["say"] = (f"Чаще всего рядом со {what} «{best[2]}» упоминается {best[1]}: {best[0]} "
                          f"{_plural(best[0], 'раз', 'раза', 'раз')} в одном предложении.")
    places = []
    if loc["names"] and has_cats:
        from app.services.geo import resolve_all

        coords = resolve_all(session, loc["names"])
        for n, row in zip(loc["names"], agg(loc)):
            c = coords.get(n) or {}
            if c.get("lat") is None or not sum(row):
                continue
            k = max(range(len(keys)), key=lambda i: row[i])
            places.append({"name": n, "lat": c["lat"], "lon": c["lon"], "n": sum(row), "cat": keys[k], "color": cat_color.get(keys[k])})
    out["places"] = places
    return out


# ------------------------------------------------------------------ 9. качество данных


def _quality(session: Session, d: Dictionary, ids: list[int], rows: list[dict]) -> dict:
    from app.nlp.syntax import SYNTAX_VERSION
    from app.services.dictionaries import duplicate_groups
    from app.services.entities import texts_without_entities

    base = [Hit.dictionary_id == d.id, Hit.text_id.in_(ids)]
    unverified = sum(1 for e in d.entries if e.ambiguous and not e.verified)
    false_hits = session.scalar(select(func.count()).select_from(Hit).where(*base, Hit.is_valid.is_(False))) or 0
    in_names = session.scalar(select(func.count()).select_from(Hit).where(*base, Hit.in_name, Hit.is_valid.is_(None))) or 0
    stale = session.scalar(select(func.count()).select_from(Hit).where(*base, Hit.syntax_version != SYNTAX_VERSION)) or 0
    no_ner = len(texts_without_entities(session, ids))
    dups = len(duplicate_groups(d))
    dead = sum(1 for r in rows if not r["count"])
    link = f"/dictionaries/{d.id}"
    items = [
        {"ok": not unverified, "text": f"Неоднозначных непроверенных слов: {unverified}", "hint": "проверьте разбор на странице словаря", "href": link},
        {"ok": not dups, "text": f"Повторов по лемме: {dups}", "hint": "«Объединить повторы» на странице словаря", "href": link},
        {"ok": not stale, "text": f"Вхождений прежней версии разбора: {stale}", "hint": "«Пересчитать вхождения»", "href": link},
        {"ok": True, "text": f"Отмечено ложными: {false_hits}", "hint": "не считаются", "href": None},
        {"ok": True, "text": f"Вхождений внутри имён персонажей: {in_names}", "hint": "не считаются (настройка словаря)", "href": None},
        {"ok": not no_ner, "text": f"Текстов без извлечённых имён: {no_ner}", "hint": "вкладка «Имена» → «Извлечь имена»", "href": None},
        {"ok": dead < max(1, len(rows)) // 2, "text": f"Слов без вхождений: {dead} из {len(rows)}", "hint": "проверьте написание и разбор", "href": link},
    ]
    bad = sum(1 for i in items if not i["ok"])
    say = "Данные в порядке." if not bad else f"Стоит проверить {bad} {_plural(bad, 'пункт', 'пункта', 'пунктов')}."
    return {"say": say, "items": items}


# ------------------------------------------------------------------ цитаты


def _quotes(session: Session, dictionary_id: int, ids: list[int], top_rows: list[dict], n: int = 6) -> list[dict]:
    """По примеру на частое слово, из разных текстов; предпочтение — предложениям около 140 знаков."""
    from app.services.labels import short_author

    out, used_texts = [], set()
    for r in top_rows:
        q = (select(Hit.sentence_id, Hit.text_id, SentenceRow.text, Hit.token_idx, TextDoc.title, TextDoc.author, TextDoc.year_from)
             .join(SentenceRow, SentenceRow.id == Hit.sentence_id).join(TextDoc, TextDoc.id == Hit.text_id)
             .where(Hit.entry_id == r["entry_id"], Hit.text_id.in_(ids), A.valid(),
                    func.length(SentenceRow.text).between(20, 320))
             .order_by(func.abs(func.length(SentenceRow.text) - 140), Hit.id).limit(40))  # ближе к 140 знакам — лучше
        cands = session.execute(q).all()
        pick = next((c for c in cands if c.text_id not in used_texts), cands[0] if cands else None)
        if pick is None:
            continue
        used_texts.add(pick.text_id)
        from app.models import TokenRow

        tok = session.execute(select(TokenRow.start_char, TokenRow.end_char)
                              .where(TokenRow.sentence_id == pick.sentence_id, TokenRow.idx == pick.token_idx)).first()
        st, en = (tok.start_char, tok.end_char) if tok else (0, 0)
        out.append({"word": r["word"], "sentence_id": pick.sentence_id, "before": pick.text[:st], "kw": pick.text[st:en],
                    "after": pick.text[en:], "source": f"{short_author(pick.author) + ' · ' if pick.author else ''}«{pick.title}»"
                                                      f"{' · ' + str(pick.year_from) if pick.year_from else ''}"})
        if len(out) >= n:
            break
    return out


def nice_ticks(lo: float, hi: float) -> list[float]:
    """Деления логарифмической оси: 1, 3, 10, 30… в пределах данных."""
    out = []
    k = math.floor(math.log10(max(lo, 1e-9)))
    while 10 ** k <= hi * 1.01:
        for m in (1, 3):
            v = m * 10 ** k
            if lo * 0.99 <= v <= hi * 1.01:
                out.append(v)
        k += 1
    return out
