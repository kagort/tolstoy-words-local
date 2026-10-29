"""Портрет словосочетания: всё о найденных конкордансом вхождениях на одной странице.

Вход — те же фильтры, что у конкорданса (`analytics.concordance_query`): синтаксический поиск
(слово + тип связи + связанное слово, «русский → мужик») или линейный (поле «Словосочетание»).
Разделы: сводка, где встречается, контексты, ключевые слова окружения, коллокаты ±5 (logDice),
синтаксис, сравнение с главным словом без этого сочетания.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models import DictionaryEntry, Hit, SentenceRow, TextDoc, TokenRelation, TokenRow
from app.nlp.tree import norm
from app.services import analytics as A
from app.services import keywords as K
from app.services.collocations import POS_GROUPS
from app.services.labels import short_author
from app.services.stats import SIGNIFICANCE_LL, ipm, log_dice, log_likelihood, log_ratio

DENSITY_BINS = 20
COLLOC_WINDOW = 5
MAX_COLLOC_SENTENCES = 20_000
SYNTAX_TOP = 8
REL_ORDER = ["predicate", "subject_of_pred", "governor", "attribute", "head", "object", "subject", "adverbial", "conj", "case"]
COMPARE_RELS = ("predicate", "subject_of_pred", "governor", "attribute", "object", "adverbial")


def nodes(session: Session, dictionary_id: int, text_ids: list[int], filters: dict) -> list[dict]:
    """Найденные вхождения (узлы портрета) в порядке текста."""
    q, _ = A.concordance_query(session, dictionary_id, text_ids, **filters)
    sub = q.subquery()
    rows = session.execute(
        select(sub.c.id, sub.c.sentence_id, sub.c.text_id, sub.c.idx, sub.c.start_char, sub.c.end_char, sub.c.entry_id,
               sub.c.title, sub.c.author, sub.c.year_from, Hit.token_id, Hit.token_idx, SentenceRow.paragraph_idx)
        .join(Hit, Hit.id == sub.c.id).join(SentenceRow, SentenceRow.id == sub.c.sentence_id)
        .order_by(sub.c.text_id, sub.c.idx, Hit.token_idx)).all()
    return [{"hit_id": r.id, "sentence_id": r.sentence_id, "text_id": r.text_id, "sent_idx": r.idx, "para_idx": r.paragraph_idx,
             "start": r.start_char, "end": r.end_char, "entry_id": r.entry_id, "token_id": r.token_id, "token_idx": r.token_idx,
             "source": f"{r.author + ', ' if r.author else ''}«{r.title}»{', ' + str(r.year_from) if r.year_from else ''}"}
            for r in rows]


def query_lemmas(filters: dict) -> set[str]:
    """Леммы самого запроса (связанное слово, «рядом с», слова словосочетания) — в окружении их не показываем."""
    out = {norm(w) for w in (filters.get("related_lemma"), filters.get("near")) if w}
    if filters.get("phrase"):
        from app.services.phrases import parse

        out |= {lemma for s in parse(filters["phrase"]) for lemma in s.get("lemmas", ())}
    return out


def title(session: Session, filters: dict, head: DictionaryEntry | None) -> str:
    """Название сочетания для заголовка: «русский → мужик», «закон * порядок»."""
    if filters.get("phrase"):
        return filters["phrase"].strip()
    word = head.surface if head else "все слова"
    if filters.get("related_lemma"):
        return f"{filters['related_lemma']} → {word}"
    if filters.get("near"):
        return f"{word} рядом с «{filters['near']}»"
    return word


# ------------------------------------------------------------------ где встречается


def _where(session: Session, ns: list[dict], text_ids: list[int]) -> dict:
    per_text = Counter(n["text_id"] for n in ns)
    texts = session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from, TextDoc.n_words)
                            .where(TextDoc.id.in_(text_ids or [-1]))).all()
    by_text = sorted(({"text_id": t.id, "title": t.title, "author": short_author(t.author) if t.author else "", "year": t.year_from, "n": per_text[t.id],
                       "ipm": round(ipm(per_text[t.id], t.n_words or 0), 1)} for t in texts if per_text[t.id]),
                     key=lambda r: (-r["n"], -r["ipm"]))
    authors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decades: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for t in texts:
        a = authors[short_author(t.author) if t.author else "без автора"]
        a[0] += per_text[t.id]
        a[1] += t.n_words or 0
        if t.year_from:
            d = decades[t.year_from // 10 * 10]
            d[0] += per_text[t.id]
            d[1] += t.n_words or 0
    by_author = sorted(({"label": k, "n": v[0], "words": v[1], "ipm": round(ipm(v[0], v[1]), 1)} for k, v in authors.items() if v[0]),
                       key=lambda r: -r["n"])
    undated = sum(per_text[t.id] for t in texts if not t.year_from)
    by_decade = [{"label": f"{k}-е", "n": v[0], "words": v[1], "ipm": round(ipm(v[0], v[1]), 1)} for k, v in sorted(decades.items())]
    # плотность по ходу текста: относительное место предложения (0 — начало, 1 — конец), DENSITY_BINS отрезков
    last = dict(session.execute(select(SentenceRow.text_id, func.max(SentenceRow.idx))
                                .where(SentenceRow.text_id.in_(list(per_text) or [-1])).group_by(SentenceRow.text_id)).all())
    density = [0] * DENSITY_BINS
    for n in ns:
        m = last.get(n["text_id"]) or 1
        density[min(DENSITY_BINS - 1, int(DENSITY_BINS * n["sent_idx"] / (m + 1)))] += 1
    mx = max(r["ipm"] for r in by_text) if by_text else 0
    return {"by_text": by_text, "by_author": by_author, "by_decade": by_decade if len(by_decade) > 1 else [], "undated": undated,
            "density": density, "density_max": max(density) if density else 0, "ipm_max": mx}


# ------------------------------------------------------------------ контексты


def _spans(session: Session, ns: list[dict], filters: dict) -> dict[int, list[tuple[int, int, str]]]:
    """Что подсветить в предложениях: само вхождение ("kw") и слово запроса ("rel") — связанное слово или словосочетание."""
    spans: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for n in ns:
        spans[n["sentence_id"]].append((n["start"], n["end"], "kw"))
    want: dict[int, set[int]] = defaultdict(set)  # sentence_id → номера токенов запроса
    if filters.get("related_lemma"):
        q = select(Hit.sentence_id, TokenRelation.related_idx).join(Hit, Hit.id == TokenRelation.hit_id).where(
            TokenRelation.hit_id.in_([n["hit_id"] for n in ns] or [-1]), TokenRelation.related_lemma == filters["related_lemma"].lower())
        if filters.get("rel_type"):
            q = q.where(TokenRelation.rel_type == filters["rel_type"])
        for sid, idx in session.execute(q):
            want[sid].add(idx)
    if filters.get("phrase"):
        from app.services import phrases as P

        positions: dict[int, set[int]] = {}
        P.match_sentences(session, [n["sentence_id"] for n in ns], P.parse(filters["phrase"]), bool(filters.get("phrase_any_order")), positions)
        for sid, idxs in positions.items():
            want[sid] |= idxs
    if want:
        cond = or_(*[and_(TokenRow.sentence_id == sid, TokenRow.idx.in_(idxs)) for sid, idxs in want.items()])
        for sid, s, e in session.execute(select(TokenRow.sentence_id, TokenRow.start_char, TokenRow.end_char).where(cond)):
            if not any(a <= s < b for a, b, _ in spans[sid]):
                spans[sid].append((s, e, "rel"))
    return spans


def marked(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    """Предложение → куски [(текст, "" | "kw" | "rel")] для подсветки в шаблоне и выгрузке."""
    out, pos = [], 0
    for s, e, kind in sorted(set(spans)):
        if s < pos:
            continue
        if s > pos:
            out.append((text[pos:s], ""))
        out.append((text[s:e], kind))
        pos = e
    if pos < len(text):
        out.append((text[pos:], ""))
    return out


def contexts(session: Session, ns: list[dict], filters: dict, unit: str = "win") -> list[dict]:
    """Контексты узлов: предложение, ±1 предложение или абзац; одинаковые контексты (два узла рядом) — один раз."""
    if not ns:
        return []
    spans = _spans(session, ns, filters)
    groups: dict[tuple, dict] = {}
    for n in ns:
        if unit == "para":
            key = (n["text_id"], "p", n["para_idx"])
        elif unit == "win":
            key = (n["text_id"], "w", n["sent_idx"])
        else:
            key = (n["text_id"], "s", n["sent_idx"])
        g = groups.setdefault(key, {"text_id": n["text_id"], "sentence_id": n["sentence_id"], "sent_idx": n["sent_idx"],
                                    "para_idx": n["para_idx"], "source": n["source"], "hits": 0})
        g["hits"] += 1
    conds = []
    for g in groups.values():
        if unit == "para":
            conds.append(and_(SentenceRow.text_id == g["text_id"], SentenceRow.paragraph_idx == g["para_idx"]))
        elif unit == "win":
            conds.append(and_(SentenceRow.text_id == g["text_id"], SentenceRow.idx.between(g["sent_idx"] - 1, g["sent_idx"] + 1)))
        else:
            conds.append(SentenceRow.id == g["sentence_id"])
    sents: dict[int, list] = defaultdict(list)
    for k in range(0, len(conds), 500):
        for r in session.execute(select(SentenceRow.id, SentenceRow.text_id, SentenceRow.idx, SentenceRow.paragraph_idx, SentenceRow.text)
                                 .where(or_(*conds[k: k + 500]))):
            sents[r.text_id].append(r)
    out = []
    for g in groups.values():
        if unit == "para":
            rows = [r for r in sents[g["text_id"]] if r.paragraph_idx == g["para_idx"]]
        elif unit == "win":
            rows = [r for r in sents[g["text_id"]] if abs(r.idx - g["sent_idx"]) <= 1]
        else:
            rows = [r for r in sents[g["text_id"]] if r.id == g["sentence_id"]]
        rows.sort(key=lambda r: r.idx)
        g["sentences"] = [{"id": r.id, "parts": marked(r.text, spans.get(r.id, [])), "node": bool(spans.get(r.id))} for r in rows]
        out.append(g)
    return out


def plain(ctx: dict) -> str:
    """Контекст одной строкой (для выгрузки): найденное слово в [квадратных скобках], слово запроса — в {фигурных}."""
    def part(t, kind):
        return f"[{t}]" if kind == "kw" else f"{{{t}}}" if kind == "rel" else t
    return " ".join("".join(part(t, k) for t, k in s["parts"]) for s in ctx["sentences"])


# ------------------------------------------------------------------ коллокаты и синтаксис


def _collocates(session: Session, ns: list[dict], totals: Counter, exclude: set[str], stop: frozenset[str], pos_set,
                top: int = 30) -> list[dict]:
    """Слова в окне ±COLLOC_WINDOW слов (без знаков препинания) от вхождения: f(xy), logDice."""
    by_sent: dict[int, list[int]] = defaultdict(list)
    for n in ns:
        by_sent[n["sentence_id"]].append(n["token_idx"])
    sids = list(by_sent)[:MAX_COLLOC_SENTENCES]
    node_tokens = {n["token_id"] for n in ns}
    fxy: Counter = Counter()
    upos_of: dict[str, Counter] = defaultdict(Counter)
    for k in range(0, len(sids), 5000):
        toks: dict[int, list] = defaultdict(list)
        for r in session.execute(select(TokenRow.id, TokenRow.sentence_id, TokenRow.idx, TokenRow.lemma_norm, TokenRow.upos)
                                 .where(TokenRow.sentence_id.in_(sids[k: k + 5000]), TokenRow.upos != "PUNCT")
                                 .order_by(TokenRow.sentence_id, TokenRow.idx)):
            toks[r.sentence_id].append(r)
        for sid, ts in toks.items():
            where = {t.idx: i for i, t in enumerate(ts)}
            seen: set[int] = set()  # слово рядом с двумя узлами одного предложения считаем один раз
            for idx in by_sent[sid]:
                i = where.get(idx)
                if i is None:
                    continue
                for j in range(max(0, i - COLLOC_WINDOW), min(len(ts), i + COLLOC_WINDOW + 1)):
                    t = ts[j]
                    if j == i or t.id in node_tokens or t.id in seen:
                        continue
                    if pos_set and t.upos not in pos_set:
                        continue
                    lemma = t.lemma_norm
                    if not lemma or lemma in exclude or lemma in stop or not any(ch.isalpha() for ch in lemma):
                        continue
                    seen.add(t.id)
                    fxy[lemma] += 1
                    upos_of[lemma][t.upos] += 1
    fx = len(ns)
    rows = [{"lemma": lemma, "pos": upos_of[lemma].most_common(1)[0][0], "count": c, "freq": totals.get(lemma, c),
             "logdice": round(log_dice(c, fx, totals.get(lemma, c)), 2)} for lemma, c in fxy.items() if c >= 2]
    rows.sort(key=lambda r: (-r["logdice"], -r["count"]))
    return rows[:top]


def _relations(session: Session, hit_ids: list[int], scopes) -> Counter:
    if not hit_ids:
        return Counter()
    out: Counter = Counter()
    for k in range(0, len(hit_ids), 20_000):
        q = (select(TokenRelation.rel_type, TokenRelation.related_lemma, func.count())
             .where(TokenRelation.hit_id.in_(hit_ids[k: k + 20_000]), TokenRelation.scope.in_(list(scopes)), TokenRelation.negated.is_(False))
             .group_by(TokenRelation.rel_type, TokenRelation.related_lemma))
        for rt, lemma, c in session.execute(q):
            out[(rt, lemma)] += c
    return out


def _syntax(rels: Counter, exclude: set[str], stop: frozenset[str]) -> list[dict]:
    grouped: dict[str, list] = defaultdict(list)
    for (rt, lemma), c in rels.items():
        if rt != "case" and (norm(lemma) in exclude or norm(lemma) in stop):
            continue
        grouped[rt].append({"lemma": lemma, "count": c})
    out = []
    for rt in sorted(grouped, key=lambda k: REL_ORDER.index(k) if k in REL_ORDER else 99):
        items = sorted(grouped[rt], key=lambda x: -x["count"])
        out.append({"rel_type": rt, "label": A.rel_label_ud(rt), "items": items[:SYNTAX_TOP], "distinct": len(items),
                    "total": sum(x["count"] for x in items)})
    return out


def _compare(rels_a: Counter, rels_b: Counter, na: int, nb: int, exclude: set[str], stop: frozenset[str], top: int = 12) -> dict:
    """Связи вхождений сочетания против связей главного слова без него: доля вхождений, G², log-ratio."""
    rows = []
    for key in set(rels_a) | set(rels_b):
        rt, lemma = key
        if rt not in COMPARE_RELS or norm(lemma) in exclude or norm(lemma) in stop:
            continue
        a, b = rels_a.get(key, 0), rels_b.get(key, 0)
        if a + b < 3:
            continue
        ll = log_likelihood(a, b, na, nb) if na and nb else 0.0
        rows.append({"rel_type": rt, "label": A.rel_label_ud(rt), "lemma": lemma, "a": a, "b": b,
                     "pct_a": round(100 * a / na, 1) if na else 0, "pct_b": round(100 * b / nb, 1) if nb else 0,
                     "ll": round(ll, 2), "log_ratio": round(log_ratio(a, b, na, nb), 2) if na and nb else None,
                     "significant": ll >= SIGNIFICANCE_LL})
    more_a = sorted((r for r in rows if r["pct_a"] > r["pct_b"]), key=lambda r: -r["ll"])[:top]
    more_b = sorted((r for r in rows if r["pct_b"] > r["pct_a"]), key=lambda r: -r["ll"])[:top]
    return {"more_a": more_a, "more_b": more_b, "n_a": na, "n_b": nb}


# ------------------------------------------------------------------ портрет


def portrait(session: Session, dictionary_id: int, text_ids: list[int], filters: dict, *, unit: str = "win", page: int = 1,
             per_page: int = 20, pos: str = "content", min_count: int = 2, stop: frozenset[str] = frozenset(),
             scopes=A.DEFAULT_SCOPES) -> dict:
    ns = nodes(session, dictionary_id, text_ids, filters)
    head_id = filters.get("entry_id") or (Counter(n["entry_id"] for n in ns).most_common(1)[0][0] if ns else None)
    head = session.get(DictionaryEntry, head_id) if head_id else None
    out = {"title": title(session, filters, head), "head": head, "n": len(ns), "unit": unit, "page": page, "per_page": per_page,
           "filters": filters}
    if not ns:
        return out
    words = A.corpus_words(session, text_ids)
    head_hits = []
    if head:
        head_hits = list(session.scalars(select(Hit.id).where(Hit.entry_id == head.id, Hit.text_id.in_(text_ids), A.valid())))
    node_ids = {n["hit_id"] for n in ns}
    out["summary"] = {"hits": len(ns), "sentences": len({n["sentence_id"] for n in ns}), "texts": len({n["text_id"] for n in ns}),
                      "texts_total": len(text_ids), "ipm": round(ipm(len(ns), words), 2), "words": words,
                      "head_total": len(head_hits),
                      "share": round(100 * sum(1 for n in ns if n["entry_id"] == head_id) / len(head_hits), 1) if head_hits else None}
    out["where"] = _where(session, ns, text_ids)
    ctx_all = contexts(session, ns, filters, unit) if len(ns) <= per_page * 5 else None
    if ctx_all is None:  # много вхождений — контексты считаем только для узлов страницы (порядок тот же)
        keys = []
        for n in ns:
            k = (n["text_id"], n["para_idx"] if unit == "para" else n["sent_idx"])
            if not keys or keys[-1] != k:
                keys.append(k)
        n_ctx = len(dict.fromkeys(keys))
        wanted = set(list(dict.fromkeys(keys))[(page - 1) * per_page: page * per_page])
        page_nodes = [n for n in ns if (n["text_id"], n["para_idx"] if unit == "para" else n["sent_idx"]) in wanted]
        out["contexts"] = contexts(session, page_nodes, filters, unit)
    else:
        n_ctx = len(ctx_all)
        out["contexts"] = ctx_all[(page - 1) * per_page: page * per_page]
    out["n_contexts"] = n_ctx
    out["pages"] = max(1, -(-n_ctx // per_page))
    exclude = query_lemmas(filters)
    pos_set = POS_GROUPS.get(pos, POS_GROUPS["content"])[1]
    totals = K._lemma_counts(session, text_ids, pos_set)
    out["keywords"] = K.context_keywords(session, sorted({n["sentence_id"] for n in ns}), text_ids, unit=unit,
                                         node_tokens=[n["token_id"] for n in ns], exclude=exclude, pos=pos,
                                         min_count=min_count, stop=stop, top=40, totals=totals)
    exclude_colloc = exclude | {norm(head.lemma)} if head else exclude
    out["collocates"] = _collocates(session, ns, totals[0], exclude_colloc, stop, pos_set)
    rels_a = _relations(session, sorted(node_ids), scopes)
    out["syntax"] = _syntax(rels_a, exclude, stop)
    narrowed = any(filters.get(k) for k in ("related_lemma", "near", "phrase", "rel_type", "role"))
    if head and narrowed:
        rest = [h for h in head_hits if h not in node_ids]
        out["compare"] = _compare(rels_a, _relations(session, rest, scopes), len(node_ids), len(rest), exclude, stop)
        out["compare"]["head"] = head.surface
    return out


def export_rows(session: Session, dictionary_id: int, text_ids: list[int], filters: dict, unit: str = "win",
                limit: int = 20_000) -> list[dict]:
    """Все контексты портрета для выгрузки (XLSX/TXT)."""
    ns = nodes(session, dictionary_id, text_ids, filters)[:limit]
    return [{"source": c["source"], "sentence_id": c["sentence_id"], "sentence_no": c["sent_idx"], "hits": c["hits"], "text": plain(c)}
            for c in contexts(session, ns, filters, unit)]
