"""Экспорт результатов: CSV, XLSX, JSON, CoNLL-U."""

from __future__ import annotations

import io
import json
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
    }
    if d:
        p["Словарь"] = d.name
        p["Настройки словаря"] = json.dumps(d.settings, ensure_ascii=False)
    for k, v in qp.items():
        p[f"параметр: {k}"] = v
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


def _ctx(db: Session, request: Request):
    qp = request.query_params
    dict_id = _int(qp.get("dictionary_id"))
    corpus_id = _int(qp.get("corpus_id"))
    return dict_id, A.corpus_text_ids(db, corpus_id)


@router.get("/frequency.{fmt}")
def export_frequency(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400, "Формат: csv, xlsx или json")
    dict_id, ids = _ctx(db, request)
    f = A.frequency(db, dict_id, ids)
    df = pd.DataFrame(f["rows"]).rename(columns={
        "word": "Слово", "lemma": "Лемма", "upos": "Часть речи", "category": "Категория", "count": "Вхождений",
        "ipm": "ipm", "sentences": "Предложений", "texts": "Текстов", "dp": "DP (неравномерность)"}).drop(columns=["entry_id"], errors="ignore")
    heat = A.frequency_by_text(db, dict_id, ids, top=200)
    df_heat = pd.DataFrame(heat["z"], index=heat["words"], columns=heat["texts"]).reset_index().rename(columns={"index": "Слово (ipm по текстам)"}) if heat["words"] else pd.DataFrame()
    params = _params(db, request) | {"Слов в корпусе": f["corpus_words"], "Текстов": f["n_texts"]}
    return _send({"Частоты": df, "По текстам (ipm)": df_heat}, fmt, "frequency", params)


@router.get("/concordance.{fmt}")
def export_concordance(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    c = A.concordance(db, dict_id, ids, entry_id=_int(qp.get("entry_id")), role=qp.get("role") or None,
                      rel_type=qp.get("rel_type") or None, related_lemma=qp.get("related") or None,
                      include_invalid=bool(qp.get("invalid")), page=1, per_page=100_000, context_chars=10_000)
    df = pd.DataFrame([{
        "Слово словаря": i["word"], "Левый контекст": i["left"], "Словоформа": i["kw"], "Правый контекст": i["right"],
        "Роль": i["role_label"], "Подтип роли": i["role_subtype"], "Источник": i["source"], "№ предложения": i["sentence_no"],
        "ID предложения": i["sentence_id"], "Проверено": {None: "", True: "верно", False: "ложное"}[i["is_valid"]],
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
    p = A.word_profile(db, entry_id, ids, scopes=scopes, top=10_000)
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
        cp = A.compare_profile(db, entry_id, ids_a, ids_b, qp.get("rel_type") or "attribute", top=10_000)
        frames["Сравнение связей"] = pd.DataFrame(cp["rows"]).rename(columns={
            "lemma": "Лемма", "count_a": "A", "count_b": "B", "per100_a": "на 100 вхожд. A", "per100_b": "на 100 вхожд. B",
            "ll": "G²", "log_ratio": "Log-ratio", "significant": "Значимо"})
    params = _params(db, request) | {"Корпус B": A.corpus_label(db, _int(qp.get("corpus_b"))),
                                     "Слов в A": c["words_a"], "Слов в B": c["words_b"]}
    return _send(frames, fmt, "compare", params)


@router.get("/structure.{fmt}")
def export_structure(fmt: str, request: Request, db: Session = Depends(get_session)):
    if fmt not in FORMATS:
        raise HTTPException(400)
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    st = A.sentence_structure(db, dict_id, ids, _int(qp.get("entry_id")))
    keys = [("n", "Предложений"), ("avg_len", "Средняя длина (слов)"), ("median_len", "Медиана длины"),
            ("stdev_len", "Ст. отклонение длины"), ("avg_depth", "Средняя глубина дерева"), ("avg_clauses", "Среднее число клауз"),
            ("direct_speech", "% прямой речи"), ("homogeneous", "% с однородными членами"),
            ("participial", "% с причастными оборотами"), ("gerund", "% с деепричастиями")]
    rows = [{"Показатель": label, "Предложения со словами словаря": st["hits"].get(k), "Все предложения корпуса": st["all"].get(k)} for k, label in keys]
    types = sorted(set(st["hits"].get("types", {})) | set(st["all"].get("types", {})))
    rows += [{"Показатель": f"% {t}", "Предложения со словами словаря": st["hits"].get("types", {}).get(t, 0),
              "Все предложения корпуса": st["all"].get("types", {}).get(t, 0)} for t in types]
    return _send({"Структура предложений": pd.DataFrame(rows)}, fmt, "structure", _params(db, request))


@router.get("/hits.conllu")
def export_hits_conllu(request: Request, db: Session = Depends(get_session)):
    qp = request.query_params
    dict_id, ids = _ctx(db, request)
    q = select(Hit.sentence_id).where(Hit.dictionary_id == dict_id, Hit.text_id.in_(ids or [-1]), A.VALID).distinct()
    if _int(qp.get("entry_id")):
        q = q.where(Hit.entry_id == _int(qp.get("entry_id")))
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


@router.get("/dictionary/{dict_id}.csv")
def export_dictionary(dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    entries = list(db.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dict_id).order_by(DictionaryEntry.id)))
    df = pd.DataFrame([{"слово": e.surface, "категория": e.category or "", "часть речи": e.upos or "",
                        "комментарий": e.comment or "", "лемма": e.lemma, "режим": e.match_mode} for e in entries])
    return Response(df.to_csv(index=False, sep=";").encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dictionary-{dict_id}.csv"'})
