"""Экспорт результатов: CSV, XLSX, JSON, CoNLL-U."""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.db import get_session
from app.models import Dictionary, DictionaryEntry, Hit, SentenceRow, TextDoc
from app.nlp.syntax import SYNTAX_VERSION
from app.services import analytics as A
from app.services.matching import load_sentences

router = APIRouter(prefix="/export")

FORMATS = {"csv", "xlsx", "json"}


def _int(v):
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _params(db: Session, request: Request) -> dict:
    qp = request.query_params
    d = db.get(Dictionary, _int(qp.get("dictionary_id"))) if qp.get("dictionary_id") else None
    p = {
        "Дата выгрузки": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Версия приложения": __version__,
        "Версия синтаксического слоя": SYNTAX_VERSION,
        "Корпус": A.corpus_label(db, _int(qp.get("corpus_id"))),
        "Выборка": _selection(db, request).describe(),
        "Стоп-слова": _stop_label(db, request),
    }
    if d:
        p["Словарь"] = d.name
        p["Настройки словаря"] = json.dumps(d.settings, ensure_ascii=False)
    for k in dict.fromkeys(qp.keys()):
        p[f"параметр: {k}"] = ", ".join(qp.getlist(k))
    return p


def _disposition(filename: str) -> str:
    """Имя файла с кириллицей по RFC 6266: латинский запасной вариант + filename* в UTF-8."""
    ascii_name = filename.encode("ascii", "ignore").decode() or "export"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def _send(frames: dict[str, pd.DataFrame], fmt: str, name: str, params: dict) -> Response:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"{name}-{stamp}.{fmt}"
    headers = {"Content-Disposition": _disposition(filename)}
    if fmt == "csv":
        df = next(iter(frames.values()))
        data = df.to_csv(index=False, sep=";").encode("utf-8-sig")
        return Response(data, media_type="text/csv; charset=utf-8", headers=headers)
    if fmt == "json":
        payload = {"parameters": params, **{k: json.loads(v.to_json(orient="records", force_ascii=False)) for k, v in frames.items()}}
        return Response(json.dumps(payload, ensure_ascii=False, indent=1), media_type="application/json", headers=headers)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet, df in frames.items():
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
            ws = xw.sheets[sheet[:31]]
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 80)
        pd.DataFrame(list(params.items()), columns=["Параметр", "Значение"]).to_excel(xw, sheet_name="Параметры", index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)


def _selection(db: Session, request: Request):
    from app.services.selection import from_params

    if not hasattr(request.state, "selection"):
        request.state.selection = from_params(db, request.query_params)
    return request.state.selection


def _stop_label(db: Session, request: Request) -> str:
    from app.services.stopwords import label

    return label(db, request.query_params.get("stop"))


def _ctx(db: Session, request: Request):
    """Словарь и итоговая выборка текстов — те же, что на странице анализа (панель «Выборка»)."""
    return _int(request.query_params.get("dictionary_id")), _selection(db, request).ids


def _words(db: Session, request: Request) -> list[int]:
    return _selection(db, request).words


@router.get("/frequency.{fmt}")
def export_frequency(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400, "Формат: csv, xlsx или json")
    dict_id, ids = _ctx(db, request)
    words = None if request.query_params.get("all_words") else (_words(db, request) or None)
    f = A.only_words(A.frequency(db, dict_id, ids), words)
    df = pd.DataFrame(f["rows"]).rename(columns={"share": "Доля среди слов таблицы, %",
        "word": "Слово", "lemma": "Лемма", "upos": "Часть речи", "category": "Категория", "count": "Вхождений",
        "ipm": "ipm", "sentences": "Предложений", "texts": "Текстов", "dp": "DP (неравномерность)"}).drop(columns=["entry_id"], errors="ignore")
    from app.services.dictionaries import no_categories

    if no_categories(db.get(Dictionary, dict_id)):
        df = df.drop(columns=["Категория"], errors="ignore")
    heat = A.frequency_by_text(db, dict_id, ids, top=200, entry_ids=words, by=_selection(db, request).unit or "word")
    df_heat = pd.DataFrame(heat["z"], index=heat["words"], columns=heat["texts_full"]).reset_index().rename(columns={"index": "Слово (ipm по текстам)"}) if heat["words"] else pd.DataFrame()
    frames = {"Частоты": df, "По текстам (ipm)": df_heat}
    from app.services.dictionaries import no_categories

    cats = None if no_categories(db.get(Dictionary, dict_id)) else A.category_frequency(
        db, dict_id, ids, entry_ids=words, dim=request.query_params.get("cat_dim") or "author")
    if cats and cats["has_categories"]:
        frames["По категориям"] = pd.DataFrame([
            {"Категория": r["category"], "Слов в категории": r["entries"], "Вхождений": r["count"], "ipm": r["ipm"], "Доля, %": r["share"],
             **{f"ipm: {g['full']}": v for g, v in zip(cats["groups"], r["groups_ipm"])}} for r in cats["rows"]])
    params = _params(db, request) | {"Слов в корпусе": f["corpus_words"], "Текстов": f["n_texts"]}
    if cats and cats["has_categories"]:
        params["Категории по группировке"] = cats["dim_label"]
    return _send(frames, fmt, "frequency", params)


@router.get("/concordance.{fmt}")
def export_concordance(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    c = A.concordance(db, dict_id, ids, entry_id=_int(qp.get("entry_id")), entry_ids=_words(db, request), role=qp.get("role") or None,
                      rel_type=qp.get("rel_type") or None, related_lemma=qp.get("related") or None,
                      near=qp.get("near") or None, window=_int(qp.get("window")) or 3,
            phrase=qp.get("phrase") or None, phrase_any_order=bool(qp.get("phrase_any")),
            names_only=bool(qp.get("names")),
                      include_invalid=bool(qp.get("invalid")), page=1, per_page=100_000, context_chars=10_000)
    df = pd.DataFrame([{
        "Слово словаря": i["word"], "Левый контекст": i["left"], "Словоформа": i["kw"], "Правый контекст": i["right"],
        "Роль": i["role_label"], "Связь UD": i["deprel"], "Часть речи UD": i["upos"], "Подтип роли": i["role_subtype"], "Источник": i["source"], "№ предложения": i["sentence_no"],
        "ID предложения": i["sentence_id"], "Словосочетание": " | ".join(i["phrase"]), "В имени": "да" if i["in_name"] else "", "Проверено": {None: "", True: "верно", False: "ложное"}[i["is_valid"]],
    } for i in c["items"]])
    return _send({"Конкорданс": df}, fmt, "concordance", _params(db, request) | {"Строк": c["total"]})


@router.get("/profile.{fmt}")
def export_profile(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    _, ids = _ctx(db, request)
    entry_id = _int(qp.get("entry_id"))
    scopes = ("direct", "shared", "uncertain") if qp.get("uncertain") else ("direct", "shared")
    p = A.word_profile(db, entry_id, ids, scopes=scopes, top=10_000, stop=_selection(db, request).stopset)
    rows = []
    for rt, block in p["relations"].items():
        for x in block["items"]:
            rows.append({"Тип связи": block["label"], "Лемма": x["lemma"], "Частота": x["count"], "Текстов": x["texts"],
                         "logDice": x["logdice"], "Пример (ID предложения)": x["example"]})
    roles = pd.DataFrame([{"Роль": r["label"], "Вхождений": r["count"]} for r in p["roles"]])
    word = p["entry"].surface if p["entry"] else ""
    return _send({"Связи": pd.DataFrame(rows), "Роли": roles}, fmt, f"profile-{word}",
                 _params(db, request) | {"Слово": word, "Вхождений": p["total"]})


@router.get("/compare.{fmt}")
def export_compare(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    dict_id, ids_a = _ctx(db, request)
    ids_b = A.corpus_text_ids(db, _int(qp.get("corpus_b")))
    c = A.compare_frequency(db, dict_id, ids_a, ids_b)
    df = pd.DataFrame(c["rows"]).drop(columns=["entry_id"], errors="ignore").rename(columns={
        "word": "Слово", "count_a": "Частота A", "count_b": "Частота B", "ipm_a": "ipm A", "ipm_b": "ipm B",
        "ll": "Log-likelihood (G²)", "log_ratio": "Log-ratio", "significant": "Значимо (p<0.05)"})
    frames = {"Сравнение частот": df}
    entry_id = _int(qp.get("entry_id"))
    if entry_id:
        cp = A.compare_profile(db, entry_id, ids_a, ids_b, qp.get("rel_type") or "attribute", top=10_000,
                               stop=_selection(db, request).stopset)
        frames["Сравнение связей"] = pd.DataFrame(cp["rows"]).rename(columns={
            "lemma": "Лемма", "count_a": "A", "count_b": "B", "per100_a": "на 100 вхожд. A", "per100_b": "на 100 вхожд. B",
            "ll": "G²", "log_ratio": "Log-ratio", "significant": "Значимо"})
    params = _params(db, request) | {"Корпус B": A.corpus_label(db, _int(qp.get("corpus_b"))),
                                     "Слов в A": c["words_a"], "Слов в B": c["words_b"]}
    return _send(frames, fmt, "compare", params)


@router.get("/overview.{fmt}")
def export_overview(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _overview_context

    dict_id, ids = _ctx(db, request)
    ov = _overview_context(db, request, dict_id, ids, all_groups=True)["ov"]
    head = {ov["dim_label"]: [g.get("full", g["label"]) for g in ov["groups"]], "Слов в группе": [g["n_words"] for g in ov["groups"]],
            "Текстов в группе": [g["n_texts"] for g in ov["groups"]]}
    frames = {}
    for metric, title in (("ipm", "ipm"), ("count", "Вхождения"), ("share", "Доля текстов, %")):
        data = dict(head)
        for w in ov["words"] + [ov["total"]]:
            data[w["word"]] = w[metric]
        frames[title] = pd.DataFrame(data)
    params = _params(db, request) | {"Группировка": ov["dim_label"], "Шаг периода (лет)": ov["step"],
                                     "Слов в выборке": ov["corpus_words"], "Текстов в выборке": ov["n_texts"]}
    return _send(frames, fmt, f"overview-{ov['dim']}", params)


@router.get("/structure.{fmt}")
def export_structure(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    st = A.sentence_structure(db, dict_id, ids, _int(qp.get("entry_id")), entry_ids=_words(db, request))
    keys = [("n", "Предложений"), ("avg_len", "Средняя длина (слов)"), ("median_len", "Медиана длины"),
            ("stdev_len", "Ст. отклонение длины"), ("avg_depth", "Средняя глубина дерева"), ("avg_clauses", "Среднее число клауз"),
            ("direct_speech", "% прямой речи"), ("homogeneous", "% с однородными членами"),
            ("participial", "% с причастными оборотами"), ("gerund", "% с деепричастиями")]
    rows = [{"Показатель": label, "Предложения со словами словаря": st["hits"].get(k), "Все предложения корпуса": st["all"].get(k)} for k, label in keys]
    types = sorted(set(st["hits"].get("types", {})) | set(st["all"].get("types", {})))
    rows += [{"Показатель": f"% {t}", "Предложения со словами словаря": st["hits"].get("types", {}).get(t, 0),
              "Все предложения корпуса": st["all"].get("types", {}).get(t, 0)} for t in types]
    return _send({"Структура предложений": pd.DataFrame(rows)}, fmt, "structure", _params(db, request))


@router.get("/report.{fmt}")
def export_report(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in ("xlsx", "json"):
        raise HTTPException(400, "Формат: xlsx или json")
    from app.web.pages import _report_context

    dict_id, ids = _ctx(db, request)
    rep = _report_context(db, request, dict_id, _selection(db, request))["rep"]
    from app.services.dictionaries import no_categories

    flat = no_categories(db.get(Dictionary, dict_id))
    frames = {"Слова": pd.DataFrame([{
        "Слово": c["word"], "Лемма": c["lemma"], **({} if flat else {"Категория": c["category"]}), "Вхождений": c["count"], "ipm": c["ipm"],
        "Предложений": c["sentences"], "Текстов": c["texts"], "DP": c["dp"], "Главная роль": c["top_role"],
        "Доля главной роли, %": c["top_role_pct"]} for c in rep["cards"]])}
    for dim, ov in rep["overviews"].items():
        data = {ov["dim_label"]: [g.get("full", g["label"]) for g in ov["groups"]], "Слов в группе": [g["n_words"] for g in ov["groups"]],
                "Текстов в группе": [g["n_texts"] for g in ov["groups"]]}
        for w in ov["words"] + [ov["total"]]:
            data[f"{w['word']} (ipm)"] = w["ipm"]
            data[f"{w['word']} (вхожд.)"] = w["count"]
        frames[f"По группировке «{ov['dim_label']}»"] = pd.DataFrame(data)
    rm = rep["role_matrix"]
    frames["Роли, %"] = pd.DataFrame([{"Слово": w["word"], **{r["label"]: w["pct"][k] for k, r in enumerate(rm["roles"])}} for w in rm["words"]])
    frames["Связи"] = pd.DataFrame([{"Слово": c["word"], "Тип связи": b["label"], "Лемма": x["lemma"], "Частота": x["count"],
                                     "Текстов": x["texts"], "logDice": x["logdice"]}
                                    for c in rep["cards"] for b in c["relations"].values() for x in b["items"]])
    frames["Примеры"] = pd.DataFrame([{"Слово": c["word"], "Предложение": i["sentence"], "Источник": i["source"], "ID предложения": i["sentence_id"]}
                                      for c in rep["cards"] for i in c["examples"]])
    if rep["structure"]:
        st = rep["structure"]
        keys = [("n", "Предложений"), ("avg_len", "Средняя длина"), ("median_len", "Медиана длины"), ("avg_depth", "Глубина дерева"),
                ("avg_clauses", "Клауз"), ("direct_speech", "% прямой речи"), ("homogeneous", "% с однородными"),
                ("participial", "% с причастными оборотами"), ("gerund", "% с деепричастиями")]
        frames["Структура предложений"] = pd.DataFrame([{"Показатель": label, "С выбранными словами": st["hits"].get(k),
                                                          "Все предложения": st["all"].get(k)} for k, label in keys])
    if rep["compare"]:
        frames["Сравнение"] = pd.DataFrame(rep["compare"]["rows"]).drop(columns=["entry_id"], errors="ignore")
    params = _params(db, request) | {"Слов в выборке": rep["corpus_words"], "Текстов в выборке": rep["n_texts"],
                                     "Вхождений выбранных слов": rep["total"]}
    return _send(frames, fmt, "report", params)


@router.get("/collocations.{fmt}")
def export_collocations(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.services.collocations import collocations
    from app.services.stats import MEASURES
    from app.web.pages import colloc_params

    entry_id = _int(request.query_params.get("entry_id"))
    if not entry_id:
        raise HTTPException(400, "Не выбрано слово (entry_id)")
    p = colloc_params(request.query_params)
    col = collocations(db, entry_id, _ctx(db, request)[1], stop=_selection(db, request).stopset, **{**p}, top=100_000)
    cols = {"lemma": "Коллокат", "rel_label": "Связь", "pos": "Часть речи", **MEASURES, "expected": "Ожидаемая частота",
            "fy": "Частота коллоката в выборке"}
    df = pd.DataFrame(col["rows"])
    if not df.empty:
        df = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    entry = db.get(DictionaryEntry, entry_id)
    params = _params(db, request) | {"Слово": entry.surface if entry else "", "Вхождений слова": col["fx"], "Слов в выборке": col["n"],
                                     "Режим": "окно ±%s" % col["window"] if col["mode"] == "window" else "синтаксические связи",
                                     "Мин. частота": col["min_count"]}
    return _send({"Коллокации": df}, fmt, f"collocations-{entry.surface if entry else entry_id}", params)


@router.get("/keywords.{fmt}")
def export_keywords(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _keywords_context

    ctx = _keywords_context(db, request, _selection(db, request))
    kw = ctx["kw"]
    names = {"lemma": "Лемма", "pos": "Часть речи", "count_a": "Частота в выборке", "count_b": "Частота в опорном",
             "ipm_a": "ipm выборки", "ipm_b": "ipm опорного", "ll": "G²", "log_ratio": "Log-ratio", "significant": "Значимо (p<0.05)"}
    frames = {"Ключевые слова": pd.DataFrame(kw["positive"]).rename(columns=names),
              "Отрицательные": pd.DataFrame(kw["negative"]).rename(columns=names)}
    if ctx["tfidf"]["texts"]:
        frames["TF-IDF по текстам"] = pd.DataFrame([{"Текст": t["title"], "Автор": t["author"], "Лемма": w["lemma"],
                                                      "Частота": w["count"], "TF-IDF×1000": w["score"]}
                                                     for t in ctx["tfidf"]["texts"] for w in t["words"]])
    params = _params(db, request) | {"Опорный корпус": ctx["kw_ref_label"], "Слов в выборке": kw["words_a"],
                                     "Слов в опорном": kw["words_b"]}
    return _send(frames, fmt, "keywords", params)


@router.get("/textmap.{fmt}")
def export_textmap(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _map_context

    tm = _map_context(db, request, _int(request.query_params.get("dictionary_id")), _selection(db, request))["tm"]
    if not tm:
        raise HTTPException(404, "Нет текста")
    head = {"Отрезок": list(range(1, tm["segments"] + 1)), "Предложения с": tm["seg_from"], "по": tm["seg_to"], "Слов в отрезке": tm["seg_words"]}
    frames = {}
    for metric, title in (("count", "Вхождения"), ("ipm", "ipm")):
        data = dict(head)
        for r in tm["summary"] + tm["rows"]:
            data[r["label"]] = r[metric]
        frames[title] = pd.DataFrame(data)
    frames["Дисперсия"] = pd.DataFrame([{"Слово": r["label"], "Вхождений": r["total"], "Отрезков со словом": r["range"],
                                         "DP": r["dp"], "D Жуйяна": r["d"]} for r in tm["summary"] + tm["rows"]])
    return _send(frames, fmt, f"textmap-{tm['text_id']}", _params(db, request) | {"Текст": tm["title"], "Отрезков": tm["segments"]})


@router.get("/entities.{fmt}")
def export_entities(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.services import entities as E

    sel = _selection(db, request)
    qp = request.query_params
    etype = qp.get("etype") if qp.get("etype") in E.ETYPES else None
    from app.web.pages import _entity_flags

    ix = E.entity_index(db, sel.ids, etype, qp.get("q") or "", top=None, flags=_entity_flags(qp),
                        hidden_only=bool(qp.get("hidden")), features=True)
    frames = {"Имена": pd.DataFrame([{"Имя": x["name"], "Класс": x["label"], "Упоминаний": x["count"], "Текстов": x["texts"],
                                      "Слитые варианты": ", ".join(x["variants"]), "Ручная правка": "да" if x["edited"] else "",
                                      "Оценка шума": x["noise"], "Почему": "; ".join(x["reasons"])}
                                     for x in ix["items"]])}
    unit = qp.get("eunit") if qp.get("eunit") in E.UNITS else "sentence"
    net = E.persons_network(db, sel.ids, unit, top=100)
    frames["Сеть персонажей"] = pd.DataFrame([{"Персона 1": e["a"], "Персона 2": e["b"], "Совместно": e["w"]} for e in net["edges"]])
    cross = qp.get("cross") if qp.get("cross") in E.ETYPES else "LOC"
    ed = E.entity_dictionary(db, sel.ids, _int(qp.get("dictionary_id")) or 0, sel.words, cross, unit, top=200, top_words=50)
    if ed["names"]:
        frames["Имена × словарь"] = pd.DataFrame(ed["z"], index=ed["names"], columns=[w["word"] for w in ed["words"]]).reset_index().rename(columns={"index": "Имя"})
    return _send(frames, fmt, "entities", _params(db, request) | {"Связь": E.UNITS[unit], "Класс для сопоставления": E.ETYPES[cross]})


@router.get("/topics.{fmt}")
def export_topics(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _topics_context

    tp = _topics_context(db, request, _selection(db, request))["tp"]
    if not tp or not tp["topics"]:
        raise HTTPException(404, "Темы не построены")
    frames = {"Темы": pd.DataFrame([{"Тема": t["id"], "Доля, %": t["share"], "Слова": ", ".join(t["words"])} for t in tp["topics"]])}
    frames["По текстам, %"] = pd.DataFrame([{"Текст": b["title"], "Фрагментов": b["chunks"],
                                              **{f"тема {t['id']}": b["dist"][j] for j, t in enumerate(tp["topics"])}} for b in tp["by_text"]])
    return _send(frames, fmt, "topics", _params(db, request) | {"Тем": tp["k"], "Фрагментов": tp["n_chunks"]})


@router.get("/stylometry.{fmt}")
def export_stylometry(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _stylo_context

    st = _stylo_context(db, request, _selection(db, request))["sty"]
    if not st.get("matrix"):
        raise HTTPException(404, st.get("error") or "Нет данных")
    labels = [u["label"] for u in st["units"]]
    frames = {"Единицы": pd.DataFrame([{"Единица": u["label"], "Автор": u["author"], "Слов": u["n_words"], "Ближайший": u["nearest"],
                                        "Автор ближайшего": u["nearest_author"]} for u in st["units"]]),
              "Расстояния": pd.DataFrame(st["matrix"], index=labels, columns=labels).reset_index().rename(columns={"index": "Единица"}),
              "Признаки (MFW)": pd.DataFrame({"Слово": st["features"]})}
    if st.get("ranking"):
        frames["Кто автор"] = pd.DataFrame(st["ranking"]["rows"]).rename(columns={"author": "Автор", "distance": "Среднее расстояние",
                                                                                  "units": "Единиц", "texts": "Текстов"})
    return _send(frames, fmt, "stylometry", _params(db, request) | {"MFW": st["mfw"], "Расстояние": st["kind"], "Единицы": st["unit"],
                                                                    "Точность ближайшего соседа, %": st.get("purity")})


@router.get("/textcmp.{fmt}")
def export_textcmp(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.web.pages import _textcmp_context

    tc = _textcmp_context(db, request, _selection(db, request))["tc"]
    if not tc:
        raise HTTPException(404)
    names = [t["title"] for t in tc["texts"]]
    frames = {"Метрики": pd.DataFrame([{"Метрика": r["label"], **dict(zip(names, r["values"]))} for r in tc["rows"]]),
              "Лексика": pd.DataFrame([{"Текст": n, "Только в этом тексте": ", ".join(u), "Ключевые слова": ", ".join(k)}
                                       for n, u, k in zip(names, tc["unique"], tc["keywords"])]),
              "Жаккар": pd.DataFrame(tc["jaccard"], index=names, columns=names).reset_index().rename(columns={"index": "Текст"})}
    if tc["delta"]:
        frames["Cosine Delta"] = pd.DataFrame(tc["delta"], index=names, columns=names).reset_index().rename(columns={"index": "Текст"})
    return _send(frames, fmt, "textcmp", _params(db, request))


@router.get("/hits.conllu")
def export_hits_conllu(request: Request, db: Session = Depends(get_session)):
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    q = select(Hit.sentence_id).where(Hit.dictionary_id == dict_id, Hit.text_id.in_(ids or [-1]), A.valid()).distinct()
    if _int(qp.get("entry_id")):
        q = q.where(Hit.entry_id == _int(qp.get("entry_id")))
    elif _words(db, request):
        q = q.where(Hit.entry_id.in_(_words(db, request)))
    sids = sorted(db.scalars(q))

    def gen():
        yield f"# exported = {datetime.now().isoformat(timespec='seconds')}\n# syntax_version = {SYNTAX_VERSION}\n\n"
        for k in range(0, len(sids), 500):
            chunk = sids[k : k + 500]
            sents = load_sentences(db, chunk)
            for sid in chunk:
                if sid in sents:
                    yield sents[sid].to_conllu(sid) + "\n"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="hits.conllu"'})


@router.get("/text/{text_id}.conllu")
def export_text_conllu(text_id: int, db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404)
    sids = list(db.scalars(select(SentenceRow.id).where(SentenceRow.text_id == text_id).order_by(SentenceRow.idx)))

    def gen():
        yield f"# newdoc id = {text_id}\n# title = {doc.title}\n# author = {doc.author or ''}\n# pipeline = {doc.pipeline_version}\n\n"
        for k in range(0, len(sids), 500):
            chunk = sids[k : k + 500]
            sents = load_sentences(db, chunk)
            for sid in chunk:
                if sid in sents:
                    yield sents[sid].to_conllu(sid) + "\n"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="text-{text_id}.conllu"'})


@router.get("/text/{text_id}/stats.{fmt}")
def export_text_stats(text_id: int, fmt: str, request: Request, db: Session = Depends(get_session)):
    from app.nlp.morph import pos_label
    from app.services.textstats import get_stats

    if fmt not in FORMATS:
        raise HTTPException(400)
    doc = db.get(TextDoc, text_id)
    st = get_stats(db, text_id) if doc else None
    if not st:
        raise HTTPException(404, "Текст не найден или ещё не разобран")
    summary = [("Токенов", st["size"]["tokens"]), ("Словоупотреблений", st["size"]["words"]),
               ("Предложений", st["size"]["sentences"]), ("Абзацев", st["size"]["paragraphs"]), ("Знаков", st["size"]["chars"]),
               ("Средняя длина слова, букв", st["size"]["avg_word_len"]), ("Средняя длина слова, слогов", st["size"]["avg_syllables"]),
               ("Слов из 4+ слогов, %", st["size"]["long_words_pct"])]
    lx = st["lexical"]
    summary += [("Словоформ (разных)", lx["n_forms"]), ("Лемм (разных)", lx["n_lemmas"]), ("TTR словоформ", lx["ttr_forms"]),
                ("TTR лемм", lx["ttr_lemmas"]), (f"MATTR (окно 500)", lx["mattr"]), ("Hapax legomena", lx["hapax"]),
                ("Hapax, % лемм", lx["hapax_pct"]), ("Dis legomena", lx["dis"]), ("K Юла", lx["yule_k"]),
                ("C Хердана", lx["herdan_c"]), ("Индекс Гиро", lx["guiraud"]), ("Лексическая плотность, %", lx["density"]),
                ("Наклон Ципфа", lx["zipf_slope"])]
    sx = st["syntax"]
    summary += [("Средняя длина предложения, слов", sx["avg_len"]), ("Медиана длины предложения", sx["median_len"]),
                ("Ст. отклонение длины", sx["stdev_len"]), ("Самое длинное предложение, слов", sx["max_len"]),
                ("Глубина дерева (средняя)", sx["avg_depth"]), ("Клауз на предложение", sx["avg_clauses"]),
                ("Средняя дистанция зависимости (MDD)", sx["mdd"]), ("% прямой речи", sx["direct_speech"]),
                ("% с однородными членами", sx["homogeneous"]), ("% с причастными оборотами", sx["participial"]),
                ("% с деепричастиями", sx["gerund"]), ("Существительные / глаголы", st["morphology"]["noun_verb"]),
                ("Прилагательные / существительные", st["morphology"]["adj_noun"]), ("Местоимения, %", st["morphology"]["pron_share"]),
                ("Удобочитаемость (Флеш–Оборнева)", st["readability"]["flesch_oborneva"])]
    frames = {"Сводка": pd.DataFrame(summary, columns=["Показатель", "Значение"])}
    from app.services.stopwords import label, resolve
    from app.services.textstats import frequency_lists

    stop = request.query_params.get("stop") or ""
    fq = frequency_lists(st, resolve(db, stop))
    for key, title in (("top_all", "Частотные леммы"), ("top_noun", "Существительные"), ("top_verb", "Глаголы"),
                       ("top_adj", "Прилагательные"), ("top_adv", "Наречия"), ("top_propn", "Имена собственные")):
        frames[title] = pd.DataFrame([{"Лемма": x["lemma"], "Часть речи": pos_label(x["pos"]), "Частота": x["count"], "ipm": x["ipm"]} for x in fq[key]])
    frames["Части речи"] = pd.DataFrame([{"Часть речи": pos_label(x["upos"]), "Количество": x["count"], "%": x["pct"]} for x in st["morphology"]["pos"]])
    frames["Граммемы"] = pd.DataFrame([{"Категория": g["title"], "Значение": f"{v['label']} ({v['value']})", "Количество": v["count"], "%": v["pct"]}
                                      for g in st["morphology"]["features"] for v in g["values"]])
    frames["Связи UD"] = pd.DataFrame([{"Связь": d["deprel"], "Количество": d["count"], "% слов": d["pct"]} for d in sx["deprels"]])
    frames["Пунктуация"] = pd.DataFrame([{"Знак": p["mark"], "Количество": p["count"], "На 1000 слов": p["per1000"]} for p in st["punctuation"]])
    dyn = st["dynamics"]
    dd = {"Часть текста": dyn["segments"], "Средняя длина предложения": dyn["avg_len"], "Глубина дерева": dyn["avg_depth"],
          "% прямой речи": dyn["direct_speech"]}
    for ds in dyn["dictionaries"]:
        dd[f"ipm: {ds['name']}"] = ds["ipm"]
    frames["Динамика"] = pd.DataFrame(dd)
    params = {"Текст": doc.title, "Автор": doc.author or "", "Год": doc.years, "Конвейер разбора": doc.pipeline_version or "",
              "Дата выгрузки": datetime.now().strftime("%Y-%m-%d %H:%M"), "Версия приложения": __version__,
              "Стоп-слова (частотные списки)": label(db, stop)}
    return _send(frames, fmt, f"text-{text_id}-stats", params)


@router.get("/dictionary/{dict_id}.csv")
def export_dictionary(dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    entries = list(db.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dict_id).order_by(DictionaryEntry.id)))
    from app.services.dictionaries import no_categories

    flat = no_categories(d)  # словарь «без категорий» выгружается простым списком слов (без колонки «категория»)
    df = pd.DataFrame([{"слово": e.surface, **({} if flat else {"категория": e.category or ""}), "часть речи": e.upos or "",
                        "комментарий": e.comment or "", "лемма": e.lemma, "режим": e.match_mode} for e in entries])
    return Response(df.to_csv(index=False, sep=";").encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dictionary-{dict_id}.csv"'})


@router.get("/geo.{fmt}")
def export_geo(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    from app.services.geo import KIND_LABELS, country_names, place_map

    g = place_map(db, _selection(db, request).ids, top=10_000)
    src = {"auto": "справочник", "manual": "вручную", "none": "не на карте", "missing": "не найдено"}
    df = pd.DataFrame([{"Место": p["name"], "Упоминаний": p["count"], "Текстов": p["texts"], "Широта": p["lat"], "Долгота": p["lon"],
                        "Вид": KIND_LABELS.get(p.get("kind") or "", p.get("kind") or ""),
                        "Страна": country_names().get(p.get("cc") or "", ""),
                        "Что нашлось": p["label"], "Источник координат": src.get(p["source"], p["source"])}
                       for p in g["points"] + g["unresolved"]])
    df_c = pd.DataFrame([{"Страна": c["name"], "Всего упоминаний": c["total"], "Сама страна": c["self"],
                          "Места": ", ".join(f"{pl['name']} ({pl['count']})" for pl in c["places"])} for c in g["countries"]])
    return _send({"Места": df, "Страны": df_c}, fmt, "geo", _params(db, request) | {"Мест на карте": len(g["points"])})


def _inline_fonts(static) -> str:
    """fonts.css со шрифтами внутри (data: URI) — автономный файл выглядит так же, как приложение, и без интернета."""
    import base64

    def data_uri(m):
        path = static / m.group(1).removeprefix("/static/")
        return "url(data:font/woff2;base64," + base64.b64encode(path.read_bytes()).decode("ascii") + ")"

    return re.sub(r"url\((/static/fonts/[^)]+\.woff2)\)", data_uri, (static / "fonts.css").read_text(encoding="utf-8"))


@router.get("/dashboard.html")
def export_dashboard(request: Request, db: Session = Depends(get_session)):
    """Дашборд одним HTML-файлом: стили, скрипты, библиотека графиков и контуры карты внутри — открывается без интернета.
    cdn=1 — облегчённый файл, библиотека графиков загружается из интернета."""
    from pathlib import Path

    import plotly

    from app.web.deps import templates
    from app.web.pages import _dashboard_data

    dash, sel, dict_id, _ = _dashboard_data(db, request)
    static = Path(__file__).resolve().parent.parent / "static"
    cdn = bool(request.query_params.get("cdn"))
    plotly_js = "" if cdn else (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    version = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").open(encoding="utf-8").read(200)
    m = re.search(r"plotly\.js v([\d.]+)", version)
    css, appjs, fonts, tpl = (static / "app.css").read_text(encoding="utf-8"), (static / "app.js").read_text(encoding="utf-8"), "", "dashboard_standalone.html"
    if dash.get("v2"):  # новый интерфейс: свои стили, шрифты (встроены в файл) и графики дашборда v2
        css += "\n" + (static / "v2.css").read_text(encoding="utf-8")
        appjs += "\n" + (static / "dash_v2.js").read_text(encoding="utf-8")
        fonts, tpl = _inline_fonts(static), "dashboard_v2_standalone.html"
    html = templates.get_template(tpl).render(
        dash=dash, css=css, appjs=appjs, fonts=fonts, standalone=True,
        topojson=(static / "topojson" / "world_110m.json").read_text(encoding="utf-8"), plotly=plotly_js, cdn=cdn,
        plotly_version=m.group(1) if m else "", app_version=__version__)
    name = re.sub(r"[^\w\-]+", "_", dash["title"]).strip("_")[:60] or "dashboard"
    return Response(html, media_type="text/html; charset=utf-8", headers={"Content-Disposition": _disposition(f"dashboard_{name}.html")})

