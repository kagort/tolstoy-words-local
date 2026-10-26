"""Обзорный дашборд по выборке (корпус, отмеченные тексты или один текст): всё главное на одной странице.

Тот же набор данных показывается в приложении (/dashboard) и сохраняется одним HTML-файлом, который открывается
без интернета (/export/dashboard.html): библиотека графиков, стили и контуры карты встраиваются в файл.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Dictionary, SentenceRow, TextDoc
from app.services import analytics as A


def _now_local() -> datetime:
    """Текущее время в часовом поясе приложения (TIMEZONE); без базы часовых поясов — UTC."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(settings.timezone))
    except Exception:  # noqa: BLE001 — нет tzdata или неверное имя пояса
        return datetime.now(timezone.utc)


def build(session: Session, sel, dictionary_id: int | None) -> dict:
    """Данные дашборда. sel — выборка (services.selection.Selection)."""
    from app.services.entities import entity_index
    from app.services.geo import place_map
    from app.services.keywords import keywords, reference_ids
    from app.services.labels import short_author
    from app.services.textstats import get_stats

    ids = sel.ids
    texts = list(session.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.year_from, TextDoc.n_words,
                                        TextDoc.n_sentences).where(TextDoc.id.in_(ids or [-1])).order_by(TextDoc.id)))
    n_words = sum(t.n_words or 0 for t in texts)
    n_sent = sum(t.n_sentences or 0 for t in texts)
    authors = sorted({t.author for t in texts if t.author})
    years = [t.year_from for t in texts if t.year_from]
    single = len(texts) == 1
    mattrs = []
    for t in texts[:200]:  # портреты считаются при разборе; для огромных выборок — по первым 200 текстам
        st = get_stats(session, t.id)
        if st and st.get("lexical", {}).get("mattr"):
            mattrs.append(st["lexical"]["mattr"])
    avg_len = round(n_words / n_sent, 1) if n_sent else 0
    out = {
        "title": (f"«{texts[0].title}»" + (f", {texts[0].author}" if texts[0].author else "")) if single else sel.corpus_name,
        "selection": sel.describe(), "generated": _now_local().strftime("%d.%m.%Y %H:%M"), "single": single,
        "tiles": [
            {"label": "текстов", "value": len(texts)},
            {"label": "авторов", "value": len(authors)},
            {"label": "слов", "value": f"{n_words:,}".replace(",", " ")},
            {"label": "предложений", "value": f"{n_sent:,}".replace(",", " ")},
            {"label": "слов в предложении (среднее)", "value": avg_len},
            {"label": "лексическое разнообразие (MATTR)", "value": round(statistics.mean(mattrs), 3) if mattrs else "—"},
        ],
        "years": f"{min(years)}–{max(years)}" if years and min(years) != max(years) else (str(years[0]) if years else ""),
        "dictionary": None,
    }
    d = session.get(Dictionary, dictionary_id) if dictionary_id else None
    if d is not None and ids:
        freq = A.frequency(session, d.id, ids)
        rows = [r for r in freq["rows"] if r["count"]]
        from app.services.dictionaries import no_categories

        cats = None if no_categories(d) else A.category_frequency(session, d.id, ids, dim="author" if len(authors) > 1 else "text")
        top_ids = [r["entry_id"] for r in rows[:5]]
        dim = "text" if single or len(authors) <= 1 else "author"
        ov = A.overview(session, d.id, ids, top_ids, dim) if top_ids and not single else None
        out["dictionary"] = {
            "name": d.name, "hits": sum(r["count"] for r in rows), "ipm": round(A.ipm(sum(r["count"] for r in rows), n_words), 1),
            "found": len(rows), "entries": len(freq["rows"]),
            "top": [{"word": r["word"], "count": r["count"], "ipm": r["ipm"]} for r in rows[:20]],
            "categories": [{"category": r["category"], "count": r["count"], "ipm": r["ipm"], "share": r["share"]}
                           for r in cats["rows"] if r["count"]][:20] if cats and cats["has_categories"] else [],
            "overview": ov,
        }
        if single:
            from app.services.textmap import text_map

            tm = text_map(session, texts[0].id, d.id, None, 50, top=8)
            out["textmap"] = {"rows": tm["rows"][:8], "segments": tm["segments"], "all": tm["summary"][0]}
    ref, ref_label = reference_ids(session, ids, None)
    kw = keywords(session, ids, ref, min_count=5, stop=sel.stopset, top=30) if ids else None
    out["keywords"] = {"items": [{"lemma": r["lemma"], "ll": r["ll"], "count": r["count_a"]} for r in kw["positive"][:30]],
                       "ref": ref_label, "has_ref": bool(ref)} if kw else None
    per = entity_index(session, ids, "PER", top=15)["items"] if ids else []
    out["persons"] = [{"name": p["name"], "count": p["count"]} for p in per]
    geo = place_map(session, ids, top=150) if ids else {"points": [], "countries": []}
    out["places"] = [{"name": p["name"], "lat": p["lat"], "lon": p["lon"], "count": p["count"], "label": p["label"]}
                     for p in geo["points"]]
    out["countries"] = [{"name": c["name"], "iso3": c["iso3"], "total": c["total"]} for c in geo["countries"] if c["iso3"]][:15]
    out["texts"] = [{"title": t.title, "author": short_author(t.author) or "—", "year": t.year_from or "", "words": t.n_words}
                    for t in texts[:50]]
    out["n_texts_hidden"] = max(0, len(texts) - 50)
    out["paragraphs"] = session.scalar(select(func.count(func.distinct(SentenceRow.paragraph_idx))).where(
        SentenceRow.text_id == texts[0].id)) if single else None
    return out
