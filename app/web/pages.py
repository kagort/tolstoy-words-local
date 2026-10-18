"""HTML-страницы и небольшие JSON-эндпоинты интерфейса."""

from __future__ import annotations

import hmac
import json
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.models import Corpus, Dictionary, DictionaryEntry, Hit, Job, SentenceRow, TextDoc, TokenRelation, TokenRow
from app.nlp.normalize import UnsupportedFormat, normalize_text
from app.nlp.syntax import SentenceAnalysis
from app.services import analytics as A
from app.services.dictionaries import (
    add_entries,
    create_dictionary,
    option_counts,
    parse_dictionary_file,
    parse_dictionary_text,
    update_entry,
)
from app.services.ingest import create_text
from app.services.metadata import lookup, normalize_source_type, parse_meta_table, parse_years
from app.services.jobs import enqueue
from app.services.stopwords import options as stop_options
from app.services.matching import load_sentences
from app.web.deps import flash, render

router = APIRouter()


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


# ------------------------------------------------------------------ вход


@router.get("/login")
def login_form(request: Request, next: str = "/"):
    return render(request, "login.html", next=next, no_password=not settings.admin_password)


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
    ok_user = hmac.compare_digest(username.strip(), settings.admin_username)
    ok_pass = bool(settings.admin_password) and hmac.compare_digest(password, settings.admin_password)
    if ok_user and ok_pass:
        request.session["user"] = username.strip()
        return redirect(next if next.startswith("/") else "/")
    flash(request, "Неверный логин или пароль", "error")
    return redirect(f"/login?next={next}")


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return redirect("/login")


# ------------------------------------------------------------------ главная


@router.get("/")
def dashboard(request: Request, db: Session = Depends(get_session)):
    stats = {
        "texts": db.scalar(select(func.count()).select_from(TextDoc)) or 0,
        "ready": db.scalar(select(func.count()).select_from(TextDoc).where(TextDoc.status == "ready")) or 0,
        "words": db.scalar(select(func.coalesce(func.sum(TextDoc.n_words), 0))) or 0,
        "sentences": db.scalar(select(func.coalesce(func.sum(TextDoc.n_sentences), 0))) or 0,
        "corpora": db.scalar(select(func.count()).select_from(Corpus)) or 0,
        "dictionaries": db.scalar(select(func.count()).select_from(Dictionary)) or 0,
        "hits": db.scalar(select(func.count()).select_from(Hit)) or 0,
    }
    jobs = list(db.scalars(select(Job).order_by(Job.id.desc()).limit(8)))
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.id)))
    corpora = list(db.scalars(select(Corpus).order_by(Corpus.id)))
    return render(request, "dashboard.html", stats=stats, jobs=jobs, dictionaries=dictionaries, corpora=corpora)


# ------------------------------------------------------------------ тексты


@router.get("/texts")
def texts_list(request: Request, corpus_id: str | None = None, db: Session = Depends(get_session)):
    cid = _int(corpus_id)
    q = select(TextDoc).order_by(TextDoc.id)
    if cid:
        q = q.where(TextDoc.corpora.any(Corpus.id == cid))
    texts = list(db.scalars(q))
    corpora = list(db.scalars(select(Corpus).order_by(Corpus.name)))
    return render(request, "texts.html", texts=texts, corpora=corpora, corpus_id=cid, max_mb=settings.upload_max_mb)


@router.post("/texts/upload")
async def texts_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    # общие сведения — для всех файлов, если для файла не указано своё
    author: str = Form(""),
    years: str = Form(""),
    genre: str = Form(""),
    source_type: str = Form("natural"),
    generator: str = Form(""),
    # сведения по каждому файлу (в порядке выбора файлов)
    f_names: list[str] = Form([]),
    f_titles: list[str] = Form([]),
    f_authors: list[str] = Form([]),
    f_years: list[str] = Form([]),
    f_genres: list[str] = Form([]),
    f_types: list[str] = Form([]),
    f_generators: list[str] = Form([]),
    meta_file: UploadFile | None = File(None),
    corpus_ids: list[str] = Form([]),
    new_corpus: str = Form(""),
    db: Session = Depends(get_session),
):
    meta: dict = {}
    if meta_file is not None and meta_file.filename:
        try:
            meta = parse_meta_table(meta_file.filename, await meta_file.read())
        except Exception as exc:  # noqa: BLE001
            flash(request, f"Таблица метаданных не прочитана: {exc}", "error")
    cids = [int(c) for c in corpus_ids if c.isdigit()]
    if new_corpus.strip():
        c = db.scalar(select(Corpus).where(Corpus.name == new_corpus.strip()))
        if c is None:
            c = Corpus(name=new_corpus.strip())
            db.add(c)
            db.flush()
        cids.append(c.id)
    # строки формы сопоставляются с файлами по имени файла (надёжнее, чем по порядку)
    per_file = {}
    for k, name in enumerate(f_names):
        def at(lst, k=k):
            return lst[k].strip() if k < len(lst) else ""
        per_file[name] = {"title": at(f_titles), "author": at(f_authors), "years": at(f_years), "genre": at(f_genres),
                          "source_type": at(f_types), "generator": at(f_generators)}
    created, errors = 0, []
    for f in [f for f in files if f.filename]:
        data = await f.read()
        if len(data) > settings.upload_max_mb * 1024 * 1024:
            errors.append(f"{f.filename}: файл больше {settings.upload_max_mb} МБ")
            continue
        row = per_file.get(f.filename, {})
        m = lookup(meta, f.filename)

        def pick(field, common="", row=row, m=m):
            return row.get(field) or m.get(field) or common

        y1, y2 = parse_years(pick("years", years))
        stype = normalize_source_type(pick("source_type")) or source_type or "natural"
        try:
            create_text(
                db, filename=f.filename, data=data, title=pick("title") or None,
                author=pick("author", author), year_from=y1, year_to=y2, genre=pick("genre", genre),
                source_type=stype, generator=pick("generator", generator), corpus_ids=cids,
            )
            created += 1
        except UnsupportedFormat as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{f.filename}: {exc}")
    db.commit()
    if created:
        flash(request, f"Загружено текстов: {created}. Разбор идёт в фоне — прогресс на странице «Задачи».", "success")
    for e in errors:
        flash(request, e, "error")
    return redirect("/texts")


@router.get("/texts/{text_id}")
def text_detail(request: Request, text_id: int, page: int = 1, db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404, "Текст не найден")
    per_page = 50
    sents = list(db.scalars(select(SentenceRow).where(SentenceRow.text_id == text_id).order_by(SentenceRow.idx)
                            .offset((page - 1) * per_page).limit(per_page)))
    corpora = list(db.scalars(select(Corpus).order_by(Corpus.name)))
    pages_n = max(1, -(-(doc.n_sentences or 0) // per_page))
    from app.services.deepparse import parse_levels

    pending = db.scalar(select(Job.id).where(Job.kind == "deepparse", Job.status.in_(["queued", "running"]),
                                            Job.params["text_id"].as_integer() == text_id))
    return render(request, "text_detail.html", doc=doc, sents=sents, corpora=corpora, page=page, pages=pages_n,
                  levels=parse_levels(db, text_id), precise_pending=bool(pending))


@router.post("/texts/{text_id}/precise")
def text_precise(request: Request, text_id: int, db: Session = Depends(get_session)):
    """Точный разбор всех предложений текста основным парсером (ступень 2 для всего текста)."""
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404)
    busy = db.scalar(select(Job.id).where(Job.kind == "deepparse", Job.status.in_(["queued", "running"]),
                                         Job.params["text_id"].as_integer() == text_id))
    if not busy:
        enqueue(db, "deepparse", {"text_id": text_id}, title=f"Точный разбор всего текста «{doc.title}»")
        db.commit()
        flash(request, "Точный разбор всего текста поставлен в очередь (прогресс — на странице «Задачи»). "
                       "После него словари пересчитаются, а портрет обновится.", "success")
    return redirect(f"/texts/{text_id}")


@router.get("/texts/{text_id}/stats")
def text_stats(request: Request, text_id: int, refresh: int = 0, stop: str = "", db: Session = Depends(get_session)):
    from app.services.stopwords import resolve
    from app.services.textstats import FEATURE_VALUE_LABELS, frequency_lists, get_stats

    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404, "Текст не найден")
    stats = get_stats(db, text_id, refresh=bool(refresh))
    if stats and stop:
        stats = {**stats, "frequency": frequency_lists(stats, resolve(db, stop))}
    return render(request, "text_stats.html", doc=doc, st=stats, FEATURE_VALUE_LABELS=FEATURE_VALUE_LABELS,
                  stop=stop, stop_options=stop_options(db))


@router.post("/texts/{text_id}/edit")
def text_edit(request: Request, text_id: int, title: str = Form(...), author: str = Form(""), years: str = Form(""),
              genre: str = Form(""), source_type: str = Form("natural"), generator: str = Form(""),
              corpus_ids: list[str] = Form([]), db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404)
    doc.title, doc.author, doc.genre = title.strip(), author.strip() or None, genre.strip() or None
    doc.year_from, doc.year_to = parse_years(years)
    doc.source_type, doc.generator = source_type, generator.strip() or None
    ids = [int(c) for c in corpus_ids if c.isdigit()]
    doc.corpora = list(db.scalars(select(Corpus).where(Corpus.id.in_(ids)))) if ids else []
    db.commit()
    flash(request, "Сведения о тексте сохранены", "success")
    return redirect(f"/texts/{text_id}")


@router.post("/texts/{text_id}/reparse")
def text_reparse(request: Request, text_id: int, db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404)
    busy = db.scalar(select(Job.id).where(Job.kind == "ingest", Job.status.in_(["queued", "running"]),
                                         Job.params["text_id"].as_integer() == text_id))
    if busy:
        flash(request, "Этот текст уже стоит в очереди на разбор или разбирается (см. «Задачи»)", "error")
        return redirect(f"/texts/{text_id}")
    doc.norm_text = normalize_text(doc.raw_text)
    doc.status = "queued"
    doc.error = None
    enqueue(db, "ingest", {"text_id": text_id}, title=f"Повторный разбор «{doc.title}»")
    db.commit()
    flash(request, "Текст поставлен на повторный разбор", "success")
    return redirect(f"/texts/{text_id}")


@router.post("/texts/{text_id}/delete")
def text_delete(request: Request, text_id: int, db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is not None:
        db.execute(delete(Hit).where(Hit.text_id == text_id))
        db.delete(doc)
        db.commit()
        flash(request, f"Текст «{doc.title}» удалён", "success")
    return redirect("/texts")


# ------------------------------------------------------------------ корпуса


@router.get("/corpora")
def corpora_list(request: Request, db: Session = Depends(get_session)):
    corpora = list(db.scalars(select(Corpus).order_by(Corpus.name)))
    sizes = {c.id: (len(c.texts), sum(t.n_words or 0 for t in c.texts)) for c in corpora}
    return render(request, "corpora.html", corpora=corpora, sizes=sizes)


@router.post("/corpora")
def corpus_create(request: Request, name: str = Form(...), description: str = Form(""), db: Session = Depends(get_session)):
    if db.scalar(select(Corpus).where(Corpus.name == name.strip())):
        flash(request, "Корпус с таким названием уже есть", "error")
        return redirect("/corpora")
    c = Corpus(name=name.strip(), description=description.strip() or None)
    db.add(c)
    db.commit()
    return redirect(f"/corpora/{c.id}")


@router.get("/corpora/{corpus_id}")
def corpus_detail(request: Request, corpus_id: int, db: Session = Depends(get_session)):
    c = db.get(Corpus, corpus_id)
    if c is None:
        raise HTTPException(404)
    all_texts = list(db.scalars(select(TextDoc).order_by(TextDoc.author, TextDoc.title)))
    member = {t.id for t in c.texts}
    return render(request, "corpus_detail.html", corpus=c, all_texts=all_texts, member=member)


@router.post("/corpora/{corpus_id}")
def corpus_update(request: Request, corpus_id: int, name: str = Form(...), description: str = Form(""),
                  text_ids: list[str] = Form([]), db: Session = Depends(get_session)):
    c = db.get(Corpus, corpus_id)
    if c is None:
        raise HTTPException(404)
    c.name, c.description = name.strip(), description.strip() or None
    ids = [int(t) for t in text_ids if t.isdigit()]
    c.texts = list(db.scalars(select(TextDoc).where(TextDoc.id.in_(ids)))) if ids else []
    db.commit()
    flash(request, "Корпус сохранён", "success")
    return redirect(f"/corpora/{corpus_id}")


@router.post("/corpora/{corpus_id}/delete")
def corpus_delete(request: Request, corpus_id: int, db: Session = Depends(get_session)):
    c = db.get(Corpus, corpus_id)
    if c:
        db.delete(c)
        db.commit()
        flash(request, "Корпус удалён (тексты остались в базе)", "success")
    return redirect("/corpora")


# ------------------------------------------------------------------ словари


@router.get("/dictionaries")
def dictionaries_list(request: Request, db: Session = Depends(get_session)):
    dicts = list(db.scalars(select(Dictionary).order_by(Dictionary.id)))
    counts = dict(db.execute(select(Hit.dictionary_id, func.count()).group_by(Hit.dictionary_id)).all())
    return render(request, "dictionaries.html", dictionaries=dicts, counts=counts)


@router.post("/dictionaries")
async def dictionary_create(request: Request, name: str = Form(...), description: str = Form(""),
                            words: str = Form(""), file: UploadFile | None = File(None),
                            include_participles: str | None = Form(None), include_gerunds: str | None = Form(None),
                            db: Session = Depends(get_session)):
    items = []
    if file is not None and file.filename:
        items += parse_dictionary_file(file.filename, await file.read())
    if words.strip():
        items += parse_dictionary_text(words)
    if not items:
        flash(request, "Словарь пуст: загрузите файл или впишите слова", "error")
        return redirect("/dictionaries")
    if db.scalar(select(Dictionary).where(Dictionary.name == name.strip())):
        flash(request, "Словарь с таким названием уже есть", "error")
        return redirect("/dictionaries")
    d = create_dictionary(db, name, items, description.strip() or None,
                          {"include_participles": bool(include_participles), "include_gerunds": bool(include_gerunds)})
    db.commit()
    flash(request, f"Словарь создан: {len(d.entries)} слов. Проверьте разборы неоднозначных слов (подсвечены).", "success")
    return redirect(f"/dictionaries/{d.id}")


@router.get("/dictionaries/{dict_id}")
def dictionary_detail(request: Request, dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    hits = dict(db.execute(select(Hit.entry_id, func.count()).where(Hit.dictionary_id == dict_id).group_by(Hit.entry_id)).all())
    opt_counts = {e.id: option_counts(db, e) for e in d.entries if e.ambiguous or not e.verified}
    pending = db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(["queued", "running"]), Job.kind == "match")) or 0
    return render(request, "dictionary_detail.html", d=d, hits=hits, opt_counts=opt_counts, pending=pending)


@router.post("/dictionaries/{dict_id}/entries/{entry_id}")
def entry_update(request: Request, dict_id: int, entry_id: int, option: str = Form(""), lemma: str = Form(""),
                 match_mode: str = Form("lemma"), category: str = Form(""), db: Session = Depends(get_session)):
    e = db.get(DictionaryEntry, entry_id)
    if e is None or e.dictionary_id != dict_id:
        raise HTTPException(404)
    new_lemma, new_upos = None, None
    if option and "|" in option:
        new_lemma, new_upos = option.split("|", 1)
    if lemma.strip():
        new_lemma = lemma.strip()
    update_entry(db, e, lemma=new_lemma, upos=new_upos if new_upos is not None else e.upos, match_mode=match_mode, category=category)
    db.commit()
    flash(request, f"Слово «{e.surface}» обновлено; пересчёт вхождений поставлен в очередь", "success")
    return redirect(f"/dictionaries/{dict_id}#e{entry_id}")


@router.post("/dictionaries/{dict_id}/entries/{entry_id}/delete")
def entry_delete(request: Request, dict_id: int, entry_id: int, db: Session = Depends(get_session)):
    e = db.get(DictionaryEntry, entry_id)
    if e is not None and e.dictionary_id == dict_id:
        db.delete(e)
        db.commit()
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/add")
async def dictionary_add(request: Request, dict_id: int, words: str = Form(""), file: UploadFile | None = File(None),
                         db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    items = parse_dictionary_text(words) if words.strip() else []
    if file is not None and file.filename:
        items += parse_dictionary_file(file.filename, await file.read())
    new = add_entries(db, d, items)
    db.commit()
    flash(request, f"Добавлено слов: {len(new)}", "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/settings")
def dictionary_settings(request: Request, dict_id: int, name: str = Form(...), description: str = Form(""),
                        include_participles: str | None = Form(None), include_gerunds: str | None = Form(None),
                        merge_propn: str | None = Form(None), db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    d.name, d.description = name.strip(), description.strip() or None
    d.settings = {"include_participles": bool(include_participles), "include_gerunds": bool(include_gerunds),
                  "merge_propn": bool(merge_propn)}
    enqueue(db, "match", {"dictionary_id": d.id}, title=f"Пересчёт словаря «{d.name}»")
    db.commit()
    flash(request, "Настройки сохранены, словарь пересчитывается", "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/rematch")
def dictionary_rematch(request: Request, dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    enqueue(db, "match", {"dictionary_id": d.id}, title=f"Пересчёт словаря «{d.name}»")
    db.commit()
    flash(request, "Пересчёт поставлен в очередь", "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/delete")
def dictionary_delete(request: Request, dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is not None:
        db.execute(delete(Hit).where(Hit.dictionary_id == dict_id))
        db.delete(d)
        db.commit()
        flash(request, f"Словарь «{d.name}» удалён", "success")
    return redirect("/dictionaries")


# ------------------------------------------------------------------ стоп-слова


@router.get("/stopwords")
def stopwords_page(request: Request, db: Session = Depends(get_session)):
    from app.models import StopList
    from app.services.stopwords import BUILTIN_NAME, builtin_words

    lists = list(db.scalars(select(StopList).order_by(StopList.name)))
    return render(request, "stopwords.html", lists=lists, builtin=builtin_words(), builtin_name=BUILTIN_NAME)


@router.get("/stopwords/builtin.txt")
def stopwords_builtin_txt():
    from fastapi.responses import PlainTextResponse

    from app.services.stopwords import builtin_words

    return PlainTextResponse("\n".join(builtin_words()) + "\n",
                             headers={"Content-Disposition": 'attachment; filename="stopwords-ru.txt"'})


async def _stop_words_from_form(words: str, file: UploadFile | None, with_builtin: bool) -> list[str]:
    from app.services.stopwords import builtin_words, parse_words

    text = words
    if file is not None and file.filename:
        raw = await file.read()
        text += "\n" + (raw.decode("utf-8-sig", errors="replace"))
    out = parse_words(text)
    if with_builtin:
        out = list(dict.fromkeys(list(builtin_words()) + out))
    return out


@router.post("/stopwords")
async def stopwords_create(request: Request, name: str = Form(...), description: str = Form(""), words: str = Form(""),
                           file: UploadFile | None = File(None), with_builtin: str | None = Form(None),
                           db: Session = Depends(get_session)):
    from app.models import StopList

    name = name.strip()
    if not name:
        flash(request, "Укажите название списка", "error")
        return redirect("/stopwords")
    if db.scalar(select(StopList).where(StopList.name == name)):
        flash(request, "Список с таким названием уже есть", "error")
        return redirect("/stopwords")
    items = await _stop_words_from_form(words, file, bool(with_builtin))
    if not items:
        flash(request, "Список пуст: впишите слова или загрузите файл", "error")
        return redirect("/stopwords")
    db.add(StopList(name=name, description=description.strip() or None, words=items))
    db.commit()
    flash(request, f"Список «{name}» создан: {len(items)} слов. Выберите его в поле «Стоп-слова» на странице анализа или портрета текста.", "success")
    return redirect("/stopwords")


@router.post("/stopwords/{list_id}")
async def stopwords_update(request: Request, list_id: int, name: str = Form(...), description: str = Form(""),
                           words: str = Form(""), db: Session = Depends(get_session)):
    from app.models import StopList

    sl = db.get(StopList, list_id)
    if sl is None:
        raise HTTPException(404)
    other = db.scalar(select(StopList).where(StopList.name == name.strip(), StopList.id != list_id))
    if other or not name.strip():
        flash(request, "Название пустое или уже занято другим списком", "error")
        return redirect("/stopwords")
    sl.name, sl.description = name.strip(), description.strip() or None
    sl.words = await _stop_words_from_form(words, None, False)
    db.commit()
    flash(request, f"Список «{sl.name}» сохранён: {len(sl.words)} слов", "success")
    return redirect("/stopwords")


@router.post("/stopwords/{list_id}/delete")
def stopwords_delete(request: Request, list_id: int, db: Session = Depends(get_session)):
    from app.models import StopList

    sl = db.get(StopList, list_id)
    if sl is not None:
        db.delete(sl)
        db.commit()
        flash(request, f"Список «{sl.name}» удалён", "success")
    return redirect("/stopwords")


# ------------------------------------------------------------------ анализ


def _analysis_context(db: Session, request: Request) -> dict:
    from app.services.selection import corpus_texts, from_params

    qp = request.query_params
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.id)))
    corpora = list(db.scalars(select(Corpus).order_by(Corpus.name)))
    dict_id = _int(qp.get("dictionary_id")) or (dictionaries[0].id if dictionaries else None)
    sel = from_params(db, qp)
    entries = list(db.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dict_id)
                              .order_by(DictionaryEntry.category, DictionaryEntry.surface))) if dict_id else []
    valid_ids = {e.id for e in entries}
    sel.words = [w for w in sel.words if w in valid_ids]  # слова другого словаря после его смены отбрасываются
    word_counts = {}
    if dict_id:
        word_counts = {r["entry_id"]: r["count"] for r in A.frequency(db, dict_id, sel.ids)["rows"]}
    categories: dict[str, list] = {}
    for e in entries:
        categories.setdefault(e.category or "", []).append(e)
    return dict(dictionaries=dictionaries, corpora=corpora, dict_id=dict_id, corpus_id=sel.corpus_id, entries=entries,
                corpus_name=sel.corpus_name, tab=qp.get("tab", "overview"), sel=sel, sel_texts=corpus_texts(db, sel),
                facets=A.facets(db, sel.corpus_ids), word_counts=word_counts, categories=categories,
                n_texts=len(sel.ids), stop_options=stop_options(db))


def _overview_params(request: Request) -> dict:
    qp = request.query_params
    return {
        "entry_ids": [int(x) for x in qp.getlist("words") if x.isdigit()],
        "dim": qp.get("dim") or "author",
        "step": _int(qp.get("step")) or 10,
        "metric": qp.get("metric") or "ipm",
    }


def _overview_context(db: Session, request: Request, dict_id: int, text_ids: list[int]) -> dict:
    """text_ids — уже отфильтрованная выборка (панель «Выборка»)."""
    p = _overview_params(request)
    entry_ids = p["entry_ids"]
    if not entry_ids:  # по умолчанию — пять самых частых слов
        entry_ids = [r["entry_id"] for r in A.frequency(db, dict_id, text_ids)["rows"] if r["count"]][:5]
    ov = A.overview(db, dict_id, text_ids, entry_ids, p["dim"], p["step"])
    return {"ov": ov, "ov_params": p, "ov_entry_ids": entry_ids, "DIMENSIONS": A.DIMENSIONS, "PERIOD_STEPS": A.PERIOD_STEPS}


def _report_context(db: Session, request: Request, dict_id: int, sel) -> dict:
    from app.services.report import build_report, default_words

    qp = request.query_params
    entry_ids = sel.words or default_words(db, dict_id, sel.ids)
    corpus_b = qp.get("report_b")
    ids_b = A.corpus_text_ids(db, _int(corpus_b)) if corpus_b not in (None, "none") else None
    step = _int(qp.get("step")) or 10
    rep = build_report(db, dict_id, sel.ids, entry_ids, step=step if step in A.PERIOD_STEPS else 10, ids_b=ids_b, stop=sel.stopset)
    return {"rep": rep, "report_b": corpus_b, "report_b_name": A.corpus_label(db, _int(corpus_b)) if ids_b is not None else "",
            "rep_default_words": not sel.words, "DIMENSIONS": A.DIMENSIONS, "PERIOD_STEPS": A.PERIOD_STEPS}


def _default_entry(db: Session, dict_id: int, sel, text_ids: list[int]) -> int | None:
    if sel.words:
        return sel.words[0]
    rows = [r for r in A.frequency(db, dict_id, text_ids)["rows"] if r["count"]]
    return rows[0]["entry_id"] if rows else None


def colloc_params(qp) -> dict:
    return {"mode": qp.get("mode") or "window", "window": _int(qp.get("window")) or 3, "pos": qp.get("pos") or "content",
            "min_count": _int(qp.get("min_count")) or 2, "sort": qp.get("sort") or "logdice"}


def _colloc_context(db: Session, request: Request, dict_id: int, sel, entry_id: int | None) -> dict:
    from app.services.collocations import MODES, POS_GROUPS, collocations, orbit_layout
    from app.services.stats import MEASURE_HELP, MEASURES

    p = colloc_params(request.query_params)
    entry_id = entry_id or _default_entry(db, dict_id, sel, sel.ids)
    col = collocations(db, entry_id, sel.ids, stop=sel.stopset, **p) if entry_id else None
    matrix = None
    if col and col["rows"]:
        keys = [k for k in MEASURES if k != "count"]
        top_rows = col["rows"][:25]
        # ранг внутри каждой меры (0 — худший, 1 — лучший) для тепловой карты «коллокат × мера»
        z = []
        for r in top_rows:
            line = []
            for k in keys:
                vals = sorted(x.get(k) for x in top_rows if x.get(k) is not None)
                v = r.get(k)
                line.append(round(vals.index(v) / max(len(vals) - 1, 1), 3) if v is not None and len(vals) > 1 else None)
            z.append(line)
        matrix = {"lemmas": [r["lemma"] + (f" ({r['rel']})" if r["rel"] else "") for r in top_rows],
                  "measures": [MEASURES[k] for k in keys], "z": z,
                  "values": [[r.get(k) for k in keys] for r in top_rows]}
    return {"col": col, "col_params": p, "entry_id": entry_id, "orbit": orbit_layout(col["rows"][:30], p["sort"]) if col else [],
            "matrix": matrix, "MEASURES": MEASURES, "MEASURE_HELP": MEASURE_HELP, "COLLOC_MODES": MODES, "POS_GROUPS": POS_GROUPS}


def _keywords_context(db: Session, request: Request, sel) -> dict:
    from app.services.collocations import POS_GROUPS
    from app.services.keywords import MAX_TFIDF_TEXTS, keywords, reference_ids, tfidf_by_text

    qp = request.query_params
    ref = qp.get("ref") or ""
    pos = qp.get("pos") or "content"
    min_count = _int(qp.get("min_count")) or 3
    ref_ids, ref_label = reference_ids(db, sel.ids, ref)
    return {"kw": keywords(db, sel.ids, ref_ids, pos=pos, min_count=min_count, stop=sel.stopset),
            "tfidf": tfidf_by_text(db, sel.ids, pos=pos, stop=sel.stopset), "kw_ref": ref, "kw_ref_label": ref_label,
            "kw_pos": pos, "kw_min": min_count, "POS_GROUPS": POS_GROUPS, "MAX_TFIDF_TEXTS": MAX_TFIDF_TEXTS}


def _map_context(db: Session, request: Request, dict_id: int, sel) -> dict:
    from app.services.textmap import SEGMENT_CHOICES, text_map

    qp = request.query_params
    texts = list(db.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.n_words)
                            .where(TextDoc.id.in_(sel.ids or [-1])).order_by(TextDoc.author, TextDoc.title)))
    text_id = _int(qp.get("text_id"))
    if text_id not in {t.id for t in texts}:
        text_id = texts[0].id if texts else None
    segments = _int(qp.get("segments")) or 50
    tm = text_map(db, text_id, dict_id, sel.words, segments) if text_id else None
    return {"tm": tm, "map_texts": texts, "map_text_id": text_id, "SEGMENT_CHOICES": SEGMENT_CHOICES}


@router.get("/api/texts/{text_id}/passage")
def text_passage(text_id: int, request: Request, db: Session = Depends(get_session)):
    """Предложения отрезка текста с подсветкой вхождений — для щелчка по карте текста."""
    from app.services.textmap import passage

    qp = request.query_params
    words = [int(x) for x in qp.getlist("words") if x.isdigit()]
    return {"sentences": passage(db, text_id, _int(qp.get("from")) or 1, _int(qp.get("to")) or 1,
                                 _int(qp.get("dictionary_id")), words)}


@router.get("/analysis")
def analysis(request: Request, db: Session = Depends(get_session)):
    ctx = _analysis_context(db, request)
    qp = request.query_params
    dict_id, tab, sel = ctx["dict_id"], ctx["tab"], ctx["sel"]
    if not dict_id:
        return render(request, "analysis.html", **ctx)
    text_ids = sel.ids
    entry_id = _int(qp.get("entry_id"))
    ctx["entry_id"] = entry_id
    scopes = ("direct", "shared", "uncertain") if qp.get("uncertain") else ("direct", "shared")
    if tab == "overview":
        ctx.update(_overview_context(db, request, dict_id, text_ids))
    elif tab == "report":
        ctx.update(_report_context(db, request, dict_id, sel))
    elif tab == "colloc":
        ctx.update(_colloc_context(db, request, dict_id, sel, entry_id))
    elif tab == "keywords":
        ctx.update(_keywords_context(db, request, sel))
    elif tab == "map":
        ctx.update(_map_context(db, request, dict_id, sel))
    elif tab == "freq":
        # отмеченные слова — только они (переключатель all_words показывает весь словарь)
        words = None if qp.get("all_words") else (sel.words or None)
        ctx["freq"] = A.only_words(A.frequency(db, dict_id, text_ids), words)
        ctx["heat"] = A.frequency_by_text(db, dict_id, text_ids, entry_ids=words)
        ctx["freq_filtered"] = bool(words)
    elif tab == "conc":
        ctx["conc"] = A.concordance(
            db, dict_id, text_ids, entry_id=entry_id, entry_ids=sel.words, role=qp.get("role") or None,
            rel_type=qp.get("rel_type") or None, related_lemma=qp.get("related") or None,
            near=qp.get("near") or None, window=_int(qp.get("window")) or 3,
            include_invalid=bool(qp.get("invalid")), page=_int(qp.get("page")) or 1,
        )
    elif tab == "profile":
        if not entry_id and ctx["entries"]:
            if sel.words:
                entry_id = sel.words[0]
            else:
                freq = A.frequency(db, dict_id, text_ids)["rows"]
                entry_id = freq[0]["entry_id"] if freq else ctx["entries"][0].id
            ctx["entry_id"] = entry_id
        if entry_id:
            ctx["profile"] = A.word_profile(db, entry_id, text_ids, scopes=scopes, include_negated=bool(qp.get("negated")),
                                            stop=sel.stopset)
            ctx["forms"] = A.forms_distribution(db, entry_id, text_ids)
    elif tab == "compare":
        corpus_b = _int(qp.get("corpus_b"))
        ctx["corpus_b"] = corpus_b
        ctx["corpus_b_name"] = A.corpus_label(db, corpus_b)
        ids_b = A.corpus_text_ids(db, corpus_b)
        ctx["cmp"] = A.compare_frequency(db, dict_id, text_ids, ids_b)
        if entry_id:
            ctx["cmp_profile"] = A.compare_profile(db, entry_id, text_ids, ids_b, qp.get("rel_type") or "attribute", scopes=scopes,
                                                   stop=sel.stopset)
    elif tab == "structure":
        ctx["structure"] = A.sentence_structure(db, dict_id, text_ids, entry_id, entry_ids=sel.words)
    return render(request, "analysis.html", **ctx)


@router.post("/api/hits/{hit_id}/valid")
async def hit_valid(hit_id: int, request: Request, db: Session = Depends(get_session)):
    body = await request.json()
    hit = db.get(Hit, hit_id)
    if hit is None:
        return JSONResponse({"error": "нет такого вхождения"}, status_code=404)
    v = body.get("valid")
    hit.is_valid = None if v is None else bool(v)
    db.commit()
    return {"ok": True, "is_valid": hit.is_valid}


# ------------------------------------------------------------------ предложение


def _context_rows(db: Session, row: SentenceRow, n: int) -> list[SentenceRow]:
    return list(db.scalars(
        select(SentenceRow)
        .where(SentenceRow.text_id == row.text_id, SentenceRow.idx.between(row.idx - n, row.idx + n))
        .order_by(SentenceRow.idx)
    ))


@router.get("/api/sentences/{sentence_id}/context")
def sentence_context(sentence_id: int, n: int = 2, db: Session = Depends(get_session)):
    """Предложение и n соседних с каждой стороны — для всплывающего окна в конкордансе."""
    row = db.get(SentenceRow, sentence_id)
    if row is None:
        return JSONResponse({"error": "нет такого предложения"}, status_code=404)
    n = max(0, min(n, 10))
    doc = db.get(TextDoc, row.text_id)
    source = f"{doc.author + ', ' if doc.author else ''}«{doc.title}»" if doc else ""
    return {
        "id": row.id,
        "source": source,
        "sentences": [
            {"id": r.id, "idx": r.idx, "paragraph": r.paragraph_idx, "text": r.text, "current": r.id == row.id}
            for r in _context_rows(db, row, n)
        ],
    }


@router.get("/go/sentence")
def go_sentence(request: Request, id: str = "", db: Session = Depends(get_session)):
    sid = _int(id.strip().lstrip("#"))
    if sid and db.get(SentenceRow, sid):
        return redirect(f"/sentences/{sid}")
    flash(request, f"Предложение с ID «{id}» не найдено", "error")
    return redirect(request.headers.get("referer") or "/analysis")



@router.get("/sentences/{sentence_id}")
def sentence_view(request: Request, sentence_id: int, token: int | None = None, db: Session = Depends(get_session)):
    row = db.get(SentenceRow, sentence_id)
    if row is None:
        raise HTTPException(404)
    sent = load_sentences(db, [sentence_id])[sentence_id]
    an = SentenceAnalysis(sent)
    doc = db.get(TextDoc, row.text_id)
    hit_tokens = set(db.scalars(select(Hit.token_idx).where(Hit.sentence_id == sentence_id)))
    focus = token or (min(hit_tokens) if hit_tokens else None)
    roles = {t.idx: an.role(t.idx) for t in sent.tokens}
    rels = an.relations(focus) if focus else []
    tree = [{"i": t.idx, "form": t.form, "lemma": t.lemma, "upos": t.upos, "head": t.head, "rel": t.deprel,
             "role": roles[t.idx].role} for t in sent.tokens]
    groups = [{"members": [sent.tok(m).form for m in g.members], "type": g.conj_type} for g in an.groups]
    # иерархическое дерево (вложенный список) — читается и для длинных предложений
    children: dict[int, list[int]] = {}
    for t in sent.tokens:
        children.setdefault(t.head, []).append(t.idx)
    tree_children = {k: sorted(v) for k, v in children.items()}
    context = _context_rows(db, row, 3)
    prev_id = db.scalar(select(SentenceRow.id).where(SentenceRow.text_id == row.text_id, SentenceRow.idx == row.idx - 1))
    next_id = db.scalar(select(SentenceRow.id).where(SentenceRow.text_id == row.text_id, SentenceRow.idx == row.idx + 1))
    precise_pending = row.parse_level == 1 and bool(db.scalar(
        select(Job.id).where(Job.kind == "deepparse", Job.status.in_(["queued", "running"]),
                             Job.params["sentence_ids"].contains([sentence_id]))))
    return render(request, "sentence.html", row=row, sent=sent, doc=doc, roles=roles, rels=rels, focus=focus,
                  precise_pending=precise_pending,
                  tree_json=json.dumps(tree, ensure_ascii=False).replace("</", "<\\/"), groups=groups, hit_tokens=hit_tokens,
                  conllu=sent.to_conllu(sentence_id), prev_id=prev_id, next_id=next_id, context=context,
                  tree_children=tree_children)


@router.post("/sentences/{sentence_id}/precise")
def sentence_precise(request: Request, sentence_id: int, db: Session = Depends(get_session)):
    row = db.get(SentenceRow, sentence_id)
    if row is None:
        raise HTTPException(404)
    if row.parse_level == 1:
        enqueue(db, "deepparse", {"sentence_ids": [sentence_id]}, title=f"Точный разбор предложения ID {sentence_id}")
        db.commit()
        flash(request, "Точный разбор поставлен в очередь — обновите страницу через минуту", "success")
    return redirect(f"/sentences/{sentence_id}")


# ------------------------------------------------------------------ задачи


@router.get("/jobs")
def jobs_page(request: Request, db: Session = Depends(get_session)):
    jobs = list(db.scalars(select(Job).order_by(Job.id.desc()).limit(100)))
    active = any(j.status in ("queued", "running") for j in jobs)
    return render(request, "jobs.html", jobs=jobs, active=active)


@router.get("/api/jobs")
def jobs_api(db: Session = Depends(get_session)):
    jobs = list(db.scalars(select(Job).where(Job.status.in_(["queued", "running"])).order_by(Job.id)))
    return {"active": [{"id": j.id, "title": j.title, "status": j.status, "progress": j.progress, "message": j.message} for j in jobs]}


@router.post("/jobs/{job_id}/retry")
def job_retry(request: Request, job_id: int, db: Session = Depends(get_session)):
    j = db.get(Job, job_id)
    if j is not None and j.status in ("error", "cancelled"):
        if j.kind == "ingest" and (j.params or {}).get("text_id"):
            doc = db.get(TextDoc, j.params["text_id"])
            if doc is None:
                flash(request, "Текст этой задачи уже удалён", "error")
                return redirect("/jobs")
            doc.status, doc.error = "queued", None
        title = j.title if j.title.endswith("(повтор)") else j.title + " (повтор)"
        enqueue(db, j.kind, j.params, title=title)
        db.delete(j)
        db.commit()
        flash(request, "Задача поставлена повторно", "success")
    return redirect("/jobs")


@router.post("/jobs/{job_id}/cancel")
def job_cancel(request: Request, job_id: int, db: Session = Depends(get_session)):
    from app.services.jobs import cancel_job

    j = db.get(Job, job_id)
    if j is not None:
        msg = cancel_job(db, j)
        db.commit()
        flash(request, msg, "success")
    return redirect("/jobs")


@router.post("/jobs/{job_id}/delete")
def job_delete(request: Request, job_id: int, db: Session = Depends(get_session)):
    from app.services.jobs import delete_jobs

    j = db.get(Job, job_id)
    if j is not None:
        if delete_jobs(db, [j]):
            flash(request, "Задача удалена", "success")
        else:
            flash(request, "Выполняющуюся задачу нельзя удалить — сначала отмените её", "error")
        db.commit()
    return redirect("/jobs")


@router.post("/jobs/clear")
def jobs_clear(request: Request, what: str = Form("finished"), db: Session = Depends(get_session)):
    """what: finished — выполненные и отменённые; errors — с ошибкой; all — все, кроме выполняющихся."""
    from app.services.jobs import delete_jobs

    statuses = {"finished": ["done", "cancelled"], "errors": ["error"], "all": ["done", "cancelled", "error", "queued"]}
    jobs = list(db.scalars(select(Job).where(Job.status.in_(statuses.get(what, ["done", "cancelled"])))))
    n = delete_jobs(db, jobs)
    db.commit()
    flash(request, f"Удалено задач: {n}", "success")
    return redirect("/jobs")


__all__ = ["router", "TokenRow", "TokenRelation"]
