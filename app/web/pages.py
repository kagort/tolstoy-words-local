"""HTML-страницы и небольшие JSON-эндпоинты интерфейса."""

from __future__ import annotations

import hmac
import json
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.models import Corpus, Dictionary, DictionaryEntry, Entity, Hit, Job, SentenceRow, TextDoc, TokenRelation, TokenRow
from app.nlp.normalize import normalize_text
from app.nlp.syntax import SYNTAX_VERSION, SentenceAnalysis
from app.services import analytics as A
from app.services.dictionaries import (
    DEFAULT_SETTINGS,
    add_entries,
    scope_corpora,
    scope_text_ids,
    sync_scope,
    create_dictionary,
    option_counts,
    parse_dictionary_file,
    parse_dictionary_text,
    update_entry,
)
from app.services.metadata import parse_years
from app.services.jobs import enqueue
from app.services.stopwords import options as stop_options
from app.services.matching import load_sentences
from app.web.deps import flash, render, ui_mode

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


@router.get("/ui/{mode}")
def ui_switch(mode: str, next: str = "/"):
    """Переключить интерфейс у себя: v2 (новый) или classic (прежний). Выбор хранится в cookie на год."""
    from app.web.deps import UI_MODES

    resp = redirect(next if next.startswith("/") and not next.startswith("//") else "/")
    if mode in UI_MODES:
        resp.set_cookie("qv_ui", mode, max_age=365 * 24 * 3600, samesite="lax", httponly=True)
    return resp


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
    by_author: str | None = Form(None),
    skip_existing: str | None = Form(None),
    db: Session = Depends(get_session),
):
    from app.services.batchupload import run_batch

    meta_upload = (meta_file.filename, await meta_file.read()) if meta_file is not None and meta_file.filename else None
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
    uploads = [(f.filename, await f.read()) for f in files if f.filename]
    rep = run_batch(db, uploads, meta_upload=meta_upload, per_file=per_file, corpus_ids=cids,
                    common={"author": author.strip(), "years": years.strip(), "genre": genre.strip(), "source_type": source_type,
                            "generator": generator.strip()},
                    by_author=bool(by_author), skip_existing=bool(skip_existing), max_mb=settings.upload_max_mb)
    db.commit()

    def short(items, n=15):
        return ", ".join(items[:n]) + (f" и ещё {len(items) - n}" if len(items) > n else "")

    if rep.created:
        flash(request, f"Загружено текстов: {rep.created}. Разбор идёт в фоне — прогресс на странице «Задачи».", "success")
    if rep.meta_name:
        flash(request, f"Таблица «{rep.meta_name}»: сведения найдены для {rep.matched} файлов.", "success" if not rep.no_meta else "info")
    if rep.no_meta:
        flash(request, f"Нет строки в таблице метаданных ({len(rep.no_meta)}): {short(rep.no_meta)} — для них взяты общие сведения.", "error")
    if rep.rows_without_file:
        flash(request, f"Строки таблицы без файла ({len(rep.rows_without_file)}): {short(rep.rows_without_file)}.", "info")
    if rep.skipped:
        flash(request, f"Пропущены уже загруженные файлы ({len(rep.skipped)}): {short(rep.skipped)}.", "info")
    if rep.author_variants:
        flash(request, "Похоже на одного автора в разных написаниях: " + "; ".join(" / ".join(g) for g in rep.author_variants[:10])
              + ". Для системы это разные авторы (корпуса, «Авторство») — исправьте имя в карточке текста или в таблице и загрузите заново.", "error")
    if rep.corpora_created:
        flash(request, f"Созданы корпуса: {short(rep.corpora_created)}.", "success")
    for e in rep.errors:
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
              short_title: str = Form(""), corpus_ids: list[str] = Form([]), db: Session = Depends(get_session)):
    doc = db.get(TextDoc, text_id)
    if doc is None:
        raise HTTPException(404)
    doc.title, doc.author, doc.genre = title.strip(), author.strip() or None, genre.strip() or None
    meta = {k: v for k, v in (doc.meta or {}).items() if k != "short_title"}
    doc.meta = {**meta, "short_title": short_title.strip()[:60]} if short_title.strip() else meta
    doc.year_from, doc.year_to = parse_years(years)
    doc.source_type, doc.generator = source_type, generator.strip() or None
    ids = [int(c) for c in corpus_ids if c.isdigit()]
    doc.corpora = list(db.scalars(select(Corpus).where(Corpus.id.in_(ids)))) if ids else []
    db.flush()
    sync_scope(db, [text_id])  # словари для выбранных корпусов: текст вошёл в область — сопоставить, вышел — убрать вхождения
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
    return _delete_texts(request, db, [text_id])


@router.post("/texts/delete")
def texts_delete(request: Request, text_ids: list[str] = Form([]), db: Session = Depends(get_session)):
    """Удаление нескольких отмеченных текстов со всеми связями."""
    return _delete_texts(request, db, [int(t) for t in text_ids if t.isdigit()])


def _delete_texts(request: Request, db: Session, ids: list[int]):
    from app.services.textdelete import request_delete

    if not ids:
        flash(request, "Не отмечено ни одного текста", "error")
        return redirect("/texts")
    job = request_delete(db, ids)
    db.commit()
    if job:
        n = len(job.params["text_ids"])
        flash(request, f"Удаление {n} текст(ов) со всеми связями поставлено в очередь — они уже не участвуют в анализе. "
                       "Большой текст удаляется до минуты (прогресс — на странице «Задачи»).", "success")
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
    before = {t.id for t in c.texts}
    c.texts = list(db.scalars(select(TextDoc).where(TextDoc.id.in_(ids)))) if ids else []
    db.flush()
    sync_scope(db, sorted(before ^ set(ids)))
    db.commit()
    flash(request, "Корпус сохранён", "success")
    return redirect(f"/corpora/{corpus_id}")


@router.post("/corpora/{corpus_id}/delete")
def corpus_delete(request: Request, corpus_id: int, with_texts: str | None = Form(None), db: Session = Depends(get_session)):
    c = db.get(Corpus, corpus_id)
    if c:
        ids = [t.id for t in c.texts]
        db.delete(c)
        if with_texts and ids:
            from app.services.textdelete import request_delete

            request_delete(db, ids)
            flash(request, f"Корпус удалён; удаление его текстов ({len(ids)}) со всеми связями поставлено в очередь", "success")
        else:
            flash(request, "Корпус удалён (тексты остались в базе)", "success")
        db.commit()
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
                            suggest: str | None = Form(None), db: Session = Depends(get_session)):
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
    skipped: list = []
    d = create_dictionary(db, name, items, description.strip() or None,
                          {"include_participles": bool(include_participles), "include_gerunds": bool(include_gerunds)},
                          skipped=skipped)
    db.commit()
    flash(request, f"Словарь создан: {len(d.entries)} слов. Проверьте разборы неоднозначных слов (подсвечены)."
          + _skipped_note(skipped), "success")
    return redirect(f"/dictionaries/{d.id}/cognates" if suggest else f"/dictionaries/{d.id}")


DICT_SORTS = {"hits": "по числу вхождений", "abc": "по алфавиту", "added": "по порядку добавления"}


def _entry_hits(db: Session, dict_id: int) -> dict[int, int]:
    """Вхождения каждого слова словаря во всех текстах (засчитанные: без ложных и слов внутри имён)."""
    return dict(db.execute(select(Hit.entry_id, func.count()).where(Hit.dictionary_id == dict_id, A.VALID).group_by(Hit.entry_id)).all())


def _sorted_entries(entries, hits: dict[int, int], sort: str = "hits") -> list:
    if sort == "abc":
        return sorted(entries, key=lambda e: e.surface.lower())
    if sort == "added":
        return sorted(entries, key=lambda e: e.id)
    return sorted(entries, key=lambda e: (-hits.get(e.id, 0), e.surface.lower()))  # частые сверху


@router.get("/dictionaries/{dict_id}")
def dictionary_detail(request: Request, dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    hits = _entry_hits(db, dict_id)
    sort = request.query_params.get("sort") if request.query_params.get("sort") in DICT_SORTS else "hits"
    entries = _sorted_entries(d.entries, hits, sort)
    # вхождения внутри имён собственных (не считаются): «любовь» в «Любовь Андреевна»
    in_names = dict(db.execute(select(Hit.entry_id, func.count()).where(Hit.dictionary_id == dict_id, Hit.in_name,
                                                                         Hit.is_valid.is_(None)).group_by(Hit.entry_id)).all())
    opt_counts = {e.id: option_counts(db, e) for e in d.entries if e.ambiguous or not e.verified}
    pending = db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(["queued", "running"]), Job.kind == "match")) or 0
    # вхождения, найденные прежней версией синтаксического слоя: связи в профиле слова неполные до пересчёта
    stale = db.scalar(select(func.count()).select_from(Hit).where(Hit.dictionary_id == dict_id, Hit.syntax_version != SYNTAX_VERSION)) or 0
    from app.services.dictionaries import duplicate_groups

    no_category = sum(1 for e in d.entries if not e.category)
    dup_groups = duplicate_groups(d)
    return render(request, "dictionary_detail.html", d=d, hits=hits, entries=entries, sort=sort, DICT_SORTS=DICT_SORTS, in_names=in_names, opt_counts=opt_counts, pending=pending, stale=stale,
                  no_category=no_category, has_categories=no_category < len(d.entries), dup_groups=dup_groups,
                  no_cats=bool((d.settings or {}).get("no_categories")),
                  all_corpora=list(db.scalars(select(Corpus).order_by(Corpus.name))), scope=set(scope_corpora(d)),
                  names_ok=set((d.settings or {}).get("names_ok_entries", [])))


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
    from app.services.dictionaries import no_categories

    flat = no_categories(e.dictionary)  # словарь «без категорий» — поле категории не показывается и не сохраняется
    changed = update_entry(db, e, lemma=new_lemma, upos=new_upos if new_upos is not None else e.upos, match_mode=match_mode,
                           category=None if flat else category)
    db.commit()
    flash(request, f"Слово «{e.surface}» сохранено и отмечено как проверенное"
          + ("; пересчёт вхождений поставлен в очередь" if changed else ""), "success")
    return redirect(f"/dictionaries/{dict_id}#e{entry_id}")


@router.post("/dictionaries/{dict_id}/entries/{entry_id}/delete")
def entry_delete(request: Request, dict_id: int, entry_id: int, db: Session = Depends(get_session)):
    e = db.get(DictionaryEntry, entry_id)
    anchor = ""
    if e is not None and e.dictionary_id == dict_id:
        # после удаления страница открывается на соседнем слове, а не в начале списка
        sort = request.query_params.get("sort") if request.query_params.get("sort") in DICT_SORTS else "hits"
        ids = [x.id for x in _sorted_entries(db.get(Dictionary, dict_id).entries, _entry_hits(db, dict_id), sort)]
        pos = ids.index(entry_id)
        nxt = ids[pos + 1] if pos + 1 < len(ids) else (ids[pos - 1] if pos else None)
        anchor = f"#e{nxt}" if nxt else ""
        surface = e.surface
        db.delete(e)
        db.commit()
        flash(request, f"«{surface}» удалено из словаря вместе с вхождениями", "success")
        if sort != "hits":
            anchor = f"?sort={sort}{anchor}"
    return redirect(f"/dictionaries/{dict_id}{anchor}")


@router.post("/dictionaries/{dict_id}/add")
async def dictionary_add(request: Request, dict_id: int, words: str = Form(""), file: UploadFile | None = File(None),
                         db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    items = parse_dictionary_text(words) if words.strip() else []
    if file is not None and file.filename:
        items += parse_dictionary_file(file.filename, await file.read())
    skipped: list = []
    new = add_entries(db, d, items, skipped=skipped)
    db.commit()
    flash(request, f"Добавлено слов: {len(new)}." + _skipped_note(skipped), "success")
    return redirect(f"/dictionaries/{dict_id}")


def _skipped_note(skipped: list) -> str:
    """Сообщение о пропущенных повторах: «благоухания» → «благоухание» (одна лемма — одна запись)."""
    if not skipped:
        return ""
    sample = ", ".join(f"«{a}» → «{b}»" for a, b in skipped[:6])
    return f" Повторы по лемме не добавлены ({len(skipped)}): {sample}{' …' if len(skipped) > 6 else ''}."


@router.post("/dictionaries/{dict_id}/merge-duplicates")
def dictionary_merge_duplicates(request: Request, dict_id: int, db: Session = Depends(get_session)):
    """Объединить записи с одной леммой (повторы и словоформы одного слова)."""
    from app.services.dictionaries import merge_duplicates

    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    n = merge_duplicates(db, d)
    db.commit()
    flash(request, f"Повторы объединены: удалено записей {n}; их формы записаны в комментарий оставшейся записи." if n
          else "Повторов по лемме нет", "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.get("/dictionaries/{dict_id}/cognates")
def dictionary_cognates(request: Request, dict_id: int, db: Session = Depends(get_session)):
    """Однокоренные слова из текстов для слов словаря (или для слов, вписанных в поле)."""
    from app.services.wordfamily import cognates

    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    words_q = (request.query_params.get("words") or "").strip()
    entries = {e.lemma: e for e in d.entries if not e.phrase}
    words = [w.strip() for w in re.split(r"[,;\n]", words_q) if w.strip()] if words_q else list(entries)
    scope = scope_text_ids(db, d)
    found = cognates(db, words, sorted(scope) if scope is not None else None,
                     exclude={e.lemma for e in d.entries} | {e.surface for e in d.entries})
    for g in found:
        e = entries.get(g["source"])
        g["category"] = e.category if e else ""
    return render(request, "dictionary_cognates.html", d=d, found=found, words_q=words_q, n_words=len(words))


@router.post("/dictionaries/{dict_id}/add-suggested")
def dictionary_add_suggested(request: Request, dict_id: int, rows: list[str] = Form([]), back: str = Form(""),
                             db: Session = Depends(get_session)):
    """Добавить отмеченные предложения (строка: слово\tкатегория\tкомментарий) с категорией исходного слова."""
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    items = []
    for r in rows:
        word, cat, comment = (r.split("\t") + ["", "", ""])[:3]
        if word.strip():
            items.append({"surface": word.strip(), "category": cat.strip() or None, "upos": None, "comment": comment.strip() or None})
    new = add_entries(db, d, items) if items else []
    db.commit()
    flash(request, f"В словарь «{d.name}» добавлено слов: {len(new)}. Вхождения ищутся в фоне (страница «Задачи»).", "success")
    return redirect(back if back.startswith("/") else f"/dictionaries/{dict_id}")


@router.get("/dictionaries/{dict_id}/categories")
def dictionary_categories(request: Request, dict_id: int, db: Session = Depends(get_session)):
    """Предложение категорий для слов без категории: группы по смыслу, правка названий и состава."""
    from app.services.wordfamily import suggest_categories

    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    k = _int(request.query_params.get("k"))
    redo = bool(request.query_params.get("all"))  # все слова заново (существующие категории заменяются)
    sug = suggest_categories(db, dict_id, k=k if k and 2 <= k <= 20 else None, only_empty=not redo)
    from app.services.dictionaries import no_categories

    return render(request, "dictionary_categories.html", d=d, sug=sug, redo=redo, no_cats=no_categories(d),
                  n_with_cat=sum(1 for e in d.entries if e.category))


@router.post("/dictionaries/{dict_id}/categories")
async def dictionary_categories_apply(request: Request, dict_id: int, db: Session = Depends(get_session)):
    from app.services.wordfamily import apply_categories

    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    form = await request.form()
    names = {int(k[5:]): str(v) for k, v in form.items() if k.startswith("name_") and k[5:].isdigit()}
    assign = {int(k[6:]): int(v) for k, v in form.items() if k.startswith("entry_") and k[6:].isdigit() and str(v).isdigit()}
    redo = form.get("all") == "1"
    n = apply_categories(db, dict_id, names, assign, only_empty=not redo)
    note = ""
    if n and (d.settings or {}).get("no_categories"):  # категории назначены — словарь снова с категориями
        d.settings = {**d.settings, "no_categories": False}
        note = "; режим «без категорий» выключен"
    db.commit()
    flash(request, f"Категории назначены: {n} слов{note}", "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/settings")
def dictionary_settings(request: Request, dict_id: int, name: str = Form(...), description: str = Form(""),
                        include_participles: str | None = Form(None), include_gerunds: str | None = Form(None),
                        merge_propn: str | None = Form(None), count_names: str | None = Form(None),
                        no_categories: str | None = Form(None),
                        scope_corpus_ids: list[str] = Form([]), db: Session = Depends(get_session)):
    from app.services.matching import flag_name_hits

    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    d.name, d.description = name.strip(), description.strip() or None
    old = dict(d.settings or {})
    new = {**old, "include_participles": bool(include_participles), "include_gerunds": bool(include_gerunds),
           "merge_propn": bool(merge_propn), "count_names": bool(count_names), "no_categories": bool(no_categories),
           "corpus_ids": sorted({int(c) for c in scope_corpus_ids if c.isdigit()})}
    d.settings = new
    cleared = 0
    if new["no_categories"]:
        from app.services.dictionaries import clear_categories

        cleared = clear_categories(d)
    match_keys = ("include_participles", "include_gerunds", "merge_propn", "corpus_ids")
    if any(old.get(k, DEFAULT_SETTINGS.get(k, [])) != new[k] for k in match_keys):
        enqueue(db, "match", {"dictionary_id": d.id}, title=f"Пересчёт словаря «{d.name}»")
        msg = "Настройки сохранены, словарь пересчитывается"
    else:  # «считать слова в составе имён» меняет только пометки — без пересчёта
        db.flush()
        flag_name_hits(db, [d.id])
        msg = "Настройки сохранены"
    if cleared:
        msg += f"; категории удалены у {cleared} слов"
    db.commit()
    flash(request, msg, "success")
    return redirect(f"/dictionaries/{dict_id}")


@router.post("/dictionaries/{dict_id}/entries/{entry_id}/names")
def entry_names_toggle(request: Request, dict_id: int, entry_id: int, db: Session = Depends(get_session)):
    """Слово-исключение: считать его вхождения и внутри имён собственных (например, «Бог»)."""
    from app.services.matching import flag_name_hits

    d = db.get(Dictionary, dict_id)
    e = db.get(DictionaryEntry, entry_id)
    if d is None or e is None or e.dictionary_id != dict_id:
        raise HTTPException(404)
    ok = set((d.settings or {}).get("names_ok_entries", []))
    ok ^= {entry_id}
    d.settings = {**(d.settings or {}), "names_ok_entries": sorted(ok)}
    db.flush()
    flag_name_hits(db, [d.id])
    db.commit()
    flash(request, f"«{e.surface}»: " + ("вхождения внутри имён тоже считаются" if entry_id in ok else "вхождения внутри имён не считаются"),
          "success")
    return redirect(f"/dictionaries/{dict_id}#e{entry_id}")


@router.post("/dictionaries/{dict_id}/scope/add")
def dictionary_scope_add(request: Request, dict_id: int, corpus_id: str = Form(""), back: str = Form("/analysis"),
                         db: Session = Depends(get_session)):
    """Добавить корпус в область словаря и сопоставить его тексты (кнопка в «Анализе»)."""
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    cid = _int(corpus_id)
    if cid and scope_corpora(d):
        d.settings = {**(d.settings or {}), "corpus_ids": sorted(set(scope_corpora(d)) | {cid})}
        db.flush()
        c = db.get(Corpus, cid)
        n = sync_scope(db, [t.id for t in c.texts] if c else [])
        db.commit()
        flash(request, f"Корпус добавлен к словарю «{d.name}»" + ("; сопоставление поставлено в очередь" if n else ""), "success")
    return redirect(back if back.startswith("/") else "/analysis")


@router.post("/dictionaries/{dict_id}/rematch")
def dictionary_rematch(request: Request, dict_id: int, db: Session = Depends(get_session)):
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    from app.services.dictionaries import clear_categories, no_categories

    cleared = clear_categories(d) if no_categories(d) else 0  # «без категорий»: старые названия категорий не остаются
    enqueue(db, "match", {"dictionary_id": d.id}, title=f"Пересчёт словаря «{d.name}»")
    db.commit()
    flash(request, "Пересчёт поставлен в очередь" + (f"; категории удалены у {cleared} слов" if cleared else ""), "success")
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
    # в выпадающих списках и внутри категорий — сначала частые слова выборки, слова без вхождений — в конце
    entries.sort(key=lambda e: (-word_counts.get(e.id, 0), e.surface.lower()))
    cur = next((d for d in dictionaries if d.id == dict_id), None)
    from app.services.dictionaries import no_categories

    no_cats = no_categories(cur)  # настройка словаря «Без категорий»: простой список слов, графики только по словам
    if no_cats:
        sel.unit = ""
    categories: dict[str, list] = {}
    for e in sorted(entries, key=lambda e: "" if no_cats else (e.category or "")):
        categories.setdefault("" if no_cats else (e.category or ""), []).append(e)
    # словарь только для выбранных корпусов, а в выборке есть другие тексты — предупреждаем
    scope_out = 0
    if cur is not None:
        from app.services.dictionaries import scope_text_ids

        scope = scope_text_ids(db, cur)
        if scope is not None:
            scope_out = sum(1 for t in sel.ids if t not in scope)
    # вхождения прежней версии синтаксического слоя: связи (профиль, коллокации, роли) — по старым правилам до пересчёта;
    # страница предложения при этом уже показывает новые — предупреждаем, чтобы расхождение не выглядело ошибкой
    stale = db.scalar(select(func.count()).select_from(Hit).where(
        Hit.dictionary_id == dict_id, Hit.text_id.in_(sel.ids or [-1]), Hit.syntax_version != SYNTAX_VERSION)) or 0 if dict_id else 0
    stale_pending = bool(stale) and bool(db.scalar(select(func.count()).select_from(Job).where(
        Job.status.in_(["queued", "running"]), Job.kind == "match")))
    return dict(dictionaries=dictionaries, corpora=corpora, dict_id=dict_id, corpus_id=sel.corpus_id, entries=entries,
                scope_out=scope_out, stale=stale, stale_pending=stale_pending,
                corpus_name=sel.corpus_name, tab=qp.get("tab", "overview"), sel=sel, sel_texts=corpus_texts(db, sel),
                facets=A.facets(db, sel.corpus_ids), word_counts=word_counts, categories=categories, no_cats=no_cats,
                n_texts=len(sel.ids), stop_options=stop_options(db))


def _overview_params(request: Request) -> dict:
    qp = request.query_params
    return {
        "entry_ids": [int(x) for x in qp.getlist("words") if x.isdigit()],
        "dim": qp.get("dim") or "author",
        "step": _int(qp.get("step")) or 10,
        "metric": qp.get("metric") or "ipm",
    }


def _overview_context(db: Session, request: Request, dict_id: int, text_ids: list[int], all_groups: bool = False) -> dict:
    """text_ids — уже отфильтрованная выборка (панель «Выборка»); all_groups — без сворачивания в «остальные» (выгрузка)."""
    p = _overview_params(request)
    entry_ids = p["entry_ids"]
    by = "cat" if request.query_params.get("unit") == "cat" else "word"
    if not entry_ids and by == "word":  # по умолчанию — пять самых частых слов (категории — самые частые категории)
        entry_ids = [r["entry_id"] for r in A.frequency(db, dict_id, text_ids)["rows"] if r["count"]][:5]
    top = None if all_groups or request.query_params.get("gtop") == "all" else (_int(request.query_params.get("gtop")) or A.GROUPS_TOP)
    ov = A.overview(db, dict_id, text_ids, entry_ids, p["dim"], p["step"], by=by, groups_top=top)
    # распределение по текстам (точка — текст) — только для страницы нового интерфейса, не для выгрузки
    spread = None if all_groups or ui_mode(request) != "v2" else A.by_text(db, dict_id, text_ids, entry_ids, p["dim"], p["step"], by=by)
    return {"ov": ov, "spread": spread, "ov_params": p, "ov_entry_ids": entry_ids, "DIMENSIONS": A.DIMENSIONS, "PERIOD_STEPS": A.PERIOD_STEPS}


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
    tm = text_map(db, text_id, dict_id, sel.words, segments, by="cat" if sel.unit == "cat" else "word") if text_id else None
    return {"tm": tm, "map_texts": texts, "map_text_id": text_id, "SEGMENT_CHOICES": SEGMENT_CHOICES}


@router.get("/api/texts/{text_id}/passage")
def text_passage(text_id: int, request: Request, db: Session = Depends(get_session)):
    """Предложения отрезка текста с подсветкой вхождений — для щелчка по карте текста."""
    from app.services.textmap import passage

    qp = request.query_params
    words = [int(x) for x in qp.getlist("words") if x.isdigit()]
    return {"sentences": passage(db, text_id, _int(qp.get("from")) or 1, _int(qp.get("to")) or 1,
                                 _int(qp.get("dictionary_id")), words)}


def _entity_flags(qp) -> set[str]:
    from app.services.entities import FLAGS

    flags = {f for f in qp.getlist("f") if f in FLAGS}
    if qp.get("nolatin"):  # старый параметр ссылок
        flags.add("nolatin")
    return flags


def _entities_context(db: Session, request: Request, dict_id: int, sel, full: bool = True) -> dict:
    """full=False — только указатель (обновление списка при вводе в поиск, без сети персонажей и графиков)."""
    from app.services import entities as E
    from app.services.dict_build import NAME_MODES

    qp = request.query_params
    etype = qp.get("etype") if qp.get("etype") in E.ETYPES else None
    unit = qp.get("eunit") if qp.get("eunit") in E.UNITS else "sentence"
    flags = _entity_flags(qp)
    hidden_only = bool(qp.get("hidden"))
    ix = E.entity_index(db, sel.ids, etype, qp.get("q") or "", flags=flags, hidden_only=hidden_only, features=True)
    cross = qp.get("cross") if qp.get("cross") in ("LOC", "PER", "ORG") else "LOC"
    sel_texts = list(db.execute(select(TextDoc.id, TextDoc.title).where(TextDoc.id.in_(sel.ids or [-1])).order_by(TextDoc.title)))
    ctx = {"ents": ix, "ent_etype": etype, "ent_unit": unit, "ent_cross": cross, "ent_flags": flags, "ent_hidden": hidden_only,
           "merge_hints": E.merge_suggestions(ix["items"]) if not (flags or hidden_only) else [], "ent_texts": sel_texts,
           "ETYPES": E.ETYPES, "ETYPE_ONE": E.ETYPE_ONE, "UNITS": E.UNITS, "ENT_FLAGS": E.FLAGS, "NOISE_SURE": E.NOISE_SURE,
           "NAME_MODES": NAME_MODES}
    if full:
        ctx["network"] = E.persons_network(db, sel.ids, unit) if ix["n_mentions"] else None
        ctx["ent_dict"] = E.entity_dictionary(db, sel.ids, dict_id, sel.words, cross, unit) if ix["n_mentions"] else None
    return ctx


@router.get("/analysis/entities-index")
def entities_index_fragment(request: Request, db: Session = Depends(get_session)):
    """Указатель имён без остальной страницы — для поиска по мере ввода."""
    from app.web.deps import templates

    ctx = _analysis_context(db, request)
    ctx.update(_entities_context(db, request, ctx["dict_id"], ctx["sel"], full=False))
    ctx["request"] = request
    return templates.TemplateResponse(request, "_entities_index.html", ctx)


@router.post("/api/entities/hide")
async def entities_hide(request: Request, db: Session = Depends(get_session)):
    """«Это не имя» для группы имён (hidden=true) или возврат скрытых (hidden=false); без перезагрузки страницы."""
    from app.services.entities import set_hidden
    from app.services.matching import flag_name_hits

    body = await request.json()
    names = [str(x) for x in body.get("names") or [] if str(x).strip()][:20000]
    if not names:
        return JSONResponse({"error": "не отмечено ни одного имени"}, status_code=400)
    tids = set_hidden(db, names, bool(body.get("hidden", True)))
    db.flush()
    if tids:  # скрытое «имя» возвращает слова словаря в частоты, возвращённое — снова исключает
        flag_name_hits(db, text_ids=tids)
    db.commit()
    return {"n": len(set(names))}


@router.post("/entities/etype")
def entities_etype(request: Request, names: list[str] = Form([]), names_all: str = Form(""), etype: str = Form(""),
                   back: str = Form("/analysis?tab=entities"), db: Session = Depends(get_session)):
    """Перенести отмеченные имена в другой класс (персона, место, организация)."""
    from app.services.entities import ETYPE_ONE, set_etype
    from app.services.matching import flag_name_hits

    names = _names_from_form(names, names_all)
    if not names:
        flash(request, "Отметьте имена в указателе", "error")
    else:
        tids = set_etype(db, names, etype or None)
        db.flush()
        if tids:
            flag_name_hits(db, text_ids=tids)
        db.commit()
        flash(request, f"Класс «{ETYPE_ONE.get(etype, 'как определила модель')}»: имён {len(names)}", "success")
    return redirect(back if back.startswith("/") else "/analysis?tab=entities")


@router.post("/entities/to-dictionary")
def entities_to_dictionary(request: Request, names: list[str] = Form([]), names_all: str = Form(""), dict_name: str = Form(""),
                           mode: str = Form("person"), back: str = Form("/analysis?tab=entities"), db: Session = Depends(get_session)):
    """Словарь из отмеченных имён указателя: персонаж — категория, варианты имени — записи."""
    from app.services import dict_build as DB
    from app.services import entities as E
    from app.services.selection import from_params
    from urllib.parse import parse_qs, urlsplit

    chosen = set(_names_from_form(names, names_all))
    if not chosen:
        flash(request, "Отметьте имена в указателе (или «отметить все найденные»)", "error")
        return redirect(back if back.startswith("/") else "/analysis?tab=entities")
    sel = from_params(db, parse_qs(urlsplit(back).query))
    items = [x for x in E.entity_index(db, sel.ids, top=None)["items"] if x["name"] in chosen]
    try:
        d = DB.build_dictionary(db, dict_name, DB.name_groups(items, mode if mode in DB.NAME_MODES else "person", E.ETYPE_ONE),
                                f"Из указателя имён ({len(items)} имён)")
    except ValueError as exc:
        db.rollback()
        flash(request, str(exc), "error")
        return redirect(back if back.startswith("/") else "/analysis?tab=entities")
    db.commit()
    flash(request, f"Словарь «{d.name}» создан из указателя имён: {len(d.entries)} записей, сопоставление поставлено в очередь", "success")
    return redirect(f"/dictionaries/{d.id}")


@router.post("/places/to-dictionary")
def places_to_dictionary(request: Request, dict_name: str = Form(""), mode: str = Form("place"), only_map: str | None = Form(None),
                         back: str = Form("/analysis?tab=geo"), db: Session = Depends(get_session)):
    """Словарь из мест выборки (вкладка «География»)."""
    from app.services import dict_build as DB
    from app.services.geo import place_map
    from app.services.selection import from_params
    from urllib.parse import parse_qs, urlsplit

    sel = from_params(db, parse_qs(urlsplit(back).query))
    gm = place_map(db, sel.ids, top=5000)
    points = gm["points"] + ([] if only_map else gm["unresolved"])
    try:
        d = DB.build_dictionary(db, dict_name, DB.place_groups(points, mode if mode in DB.PLACE_MODES else "place"),
                                f"Из мест на карте ({len(points)} мест)")
    except ValueError as exc:
        db.rollback()
        flash(request, str(exc), "error")
        return redirect(back if back.startswith("/") else "/analysis?tab=geo")
    db.commit()
    flash(request, f"Словарь «{d.name}» создан из мест: {len(d.entries)} записей, сопоставление поставлено в очередь", "success")
    return redirect(f"/dictionaries/{d.id}")


def _names_from_form(names: list[str], names_all: str) -> list[str]:
    """Отмеченные флажки плюс «Отметить все найденные» (список всех имён по фильтру, по строке на имя)."""
    out = {n.strip() for n in names if n.strip()} | {n.strip() for n in names_all.split("\n") if n.strip()}
    return sorted(out)


def _semantic_context(db: Session, request: Request, dict_id: int, sel, entry_id: int | None) -> dict:
    from app.services import semantic as S

    qp = request.query_params
    entries = list(db.scalars(select(DictionaryEntry).where(DictionaryEntry.dictionary_id == dict_id)))
    by_id = {e.id: e for e in entries}
    seed_text = (qp.get("seed") or "").strip()
    if seed_text:
        seeds = [w.strip() for w in seed_text.replace(";", ",").replace("\n", ",").split(",") if w.strip()]
        seed_label = ", ".join(seeds)
    elif sel.words:
        seeds = [by_id[w].lemma for w in sel.words if w in by_id]
        seed_label = "выбранные слова: " + ", ".join(by_id[w].surface for w in sel.words if w in by_id)
    else:
        seeds = [e.lemma for e in entries]
        seed_label = "весь словарь"
    sim = S.similar_words(db, seeds, sel.ids, exclude={e.lemma for e in entries} | {e.surface for e in entries},
                          only_in_corpus=not qp.get("any"))
    k_groups = _int(qp.get("groups"))
    entry_id = entry_id or _default_entry(db, dict_id, sel, sel.ids)
    k = max(2, min(_int(qp.get("k")) or 3, 8))
    return {"sim": sim, "seed_text": seed_text, "seed_label": seed_label,
            "smap": S.dictionary_map(db, dict_id, k_groups if k_groups and 2 <= k_groups <= 12 else None),
            "senses": S.senses(db, entry_id, sel.ids, k=k, stop=sel.stopset) if entry_id else None,
            "entry_id": entry_id, "sense_k": k}


def _topics_context(db: Session, request: Request, sel) -> dict:
    from app.services import semantic as S

    k = max(3, min(_int(request.query_params.get("k")) or 8, 20))
    words = A.corpus_words(db, sel.ids)
    if words > 3_000_000:
        return {"tp": None, "tp_k": k, "tp_too_big": words}
    return {"tp": S.topics(db, sel.ids, k=k, stop=sel.stopset), "tp_k": k, "tp_too_big": 0}


@router.post("/dictionaries/{dict_id}/add-words")
def dictionary_add_words(request: Request, dict_id: int, words: list[str] = Form([]), back: str = Form(""),
                         db: Session = Depends(get_session)):
    """Добавить в словарь слова, отмеченные среди «похожих по смыслу»."""
    d = db.get(Dictionary, dict_id)
    if d is None:
        raise HTTPException(404)
    items = parse_dictionary_text(", ".join(w for w in words if w.strip()))
    new = add_entries(db, d, items) if items else []
    db.commit()
    flash(request, f"В словарь «{d.name}» добавлено слов: {len(new)}. Вхождения ищутся в фоне (страница «Задачи»).", "success")
    return redirect(back if back.startswith("/") else f"/dictionaries/{dict_id}")


def _stylo_context(db: Session, request: Request, sel) -> dict:
    from app.services import stylometry as ST

    qp = request.query_params
    unit = qp.get("sunit") if qp.get("sunit") in ("text", "chunk") else ("chunk" if len(sel.ids) < 4 else "text")
    chunk = _int(qp.get("chunk")) or 5000
    chunk = chunk if chunk in (2000, 5000, 10000, 20000) else 5000
    target = _int(qp.get("target"))
    res = ST.analyze(db, sel.ids, mfw=_int(qp.get("mfw")) or 100, kind=qp.get("kind") or "cosine", unit=unit, chunk=chunk,
                     target=target if target in sel.ids else None, exclude=sel.names)
    texts = list(db.execute(select(TextDoc.id, TextDoc.title, TextDoc.author).where(TextDoc.id.in_(sel.ids or [-1]))
                            .order_by(TextDoc.author, TextDoc.title)))
    return {"sty": res, "sty_texts": texts, "sty_target": target, "MFW_CHOICES": ST.MFW_CHOICES, "DISTANCES": ST.DISTANCES,
            "STY_MIN_WORDS": ST.MIN_WORDS}


def _textcmp_context(db: Session, request: Request, sel) -> dict:
    from app.services.textcompare import compare

    chosen = [int(x) for x in request.query_params.getlist("cmp") if x.isdigit() and int(x) in sel.ids][:5]
    if not chosen:
        chosen = sel.ids[:4]
    texts = list(db.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.n_words).where(TextDoc.id.in_(sel.ids or [-1]))
                            .order_by(TextDoc.author, TextDoc.title)))
    return {"tc": compare(db, chosen, sel.stopset) if len(chosen) >= 1 else None, "tc_texts": texts, "tc_chosen": chosen}


@router.get("/api/entities/mentions")
def entity_mentions(request: Request, db: Session = Depends(get_session)):
    from app.services.entities import mentions
    from app.services.selection import from_params

    sel = from_params(db, request.query_params)
    return mentions(db, sel.ids, request.query_params.get("name", ""))


@router.get("/api/entities/cooccurrence")
def entity_cooccurrence(request: Request, db: Session = Depends(get_session)):
    from app.services.entities import cooccurrence_examples
    from app.services.selection import from_params

    qp = request.query_params
    sel = from_params(db, qp)
    return {"sentences": cooccurrence_examples(db, sel.ids, qp.get("name", ""), _int(qp.get("entry")) or 0,
                                               qp.get("eunit") or "sentence")}


@router.post("/entities/override")
def entity_override(request: Request, norm: str = Form(...), etype: str = Form(""), canonical: str = Form(""),
                    hidden: str | None = Form(None), back: str = Form("/analysis?tab=entities"), db: Session = Depends(get_session)):
    from app.services.entities import set_override

    set_override(db, norm.strip(), etype or None, canonical, bool(hidden))
    db.flush()
    from app.services.matching import flag_name_hits

    # «это не имя» возвращает в частоты слова словаря внутри этого «имени»
    tids = list(db.scalars(select(func.distinct(Entity.text_id)).where(Entity.norm == norm.strip())))
    if tids:
        flag_name_hits(db, text_ids=tids)
    db.commit()
    flash(request, f"Правка для «{norm}» сохранена" + (f": объединено с «{canonical.strip()}»" if canonical.strip() else ""), "success")
    return redirect(back if back.startswith("/") else "/analysis?tab=entities")


@router.post("/entities/merge")
def entities_merge(request: Request, names: list[str] = Form([]), names_all: str = Form(""), canonical: str = Form(""),
                   scope: str = Form("all"), back: str = Form("/analysis?tab=entities"), db: Session = Depends(get_session)):
    """Объединить отмеченные имена в одного персонажа — во всех текстах или только в одном."""
    from app.services.entities import merge_names

    names = _names_from_form(names, names_all)

    target = canonical.strip() or (max(names, key=len) if names else "")
    if len(set(names) | {target}) < 2 or not target:
        flash(request, "Отметьте хотя бы два имени (или одно и впишите имя персонажа)", "error")
    else:
        text_id = _int(scope) if scope != "all" else None
        n = merge_names(db, names, target, text_id)
        db.commit()
        where = f"в тексте «{db.get(TextDoc, text_id).title}»" if text_id and db.get(TextDoc, text_id) else "во всех текстах"
        flash(request, f"Объединено в «{target}» {where}: вариантов {n}", "success")
    return redirect(back if back.startswith("/") else "/analysis?tab=entities")


@router.post("/entities/split")
def entities_split(request: Request, variant: str = Form(...), text_ids: list[str] = Form([]),
                   back: str = Form("/analysis?tab=entities"), db: Session = Depends(get_session)):
    from app.services.entities import split_name

    split_name(db, variant.strip(), [int(t) for t in text_ids if t.isdigit()])
    db.commit()
    flash(request, f"«{variant}» отделено от персонажа", "success")
    return redirect(back if back.startswith("/") else "/analysis?tab=entities")


@router.post("/places")
def place_edit(request: Request, name: str = Form(...), action: str = Form("coords"), coords: str = Form(""),
               option: str = Form(""), kind: str = Form(""), back: str = Form("/analysis?tab=geo"),
               db: Session = Depends(get_session)):
    """Ручная правка места на карте: координаты, вариант из справочника, «не на карте», сброс."""
    from app.services.geo import set_place

    try:
        msg = set_place(db, name, action, coords, _int(option), kind)
        db.commit()
        flash(request, msg, "success")
    except ValueError as exc:
        db.rollback()
        flash(request, str(exc), "error")
    return redirect(back if back.startswith("/") else "/analysis?tab=geo")


@router.post("/entities/extract")
def entities_extract(request: Request, text_ids: list[str] = Form([]), back: str = Form("/analysis?tab=entities"),
                     db: Session = Depends(get_session)):
    """Поставить в очередь извлечение имён для текстов без разметки (тексты, загруженные до версии 0.6)."""
    n = 0
    for tid in [int(t) for t in text_ids if t.isdigit()]:
        doc = db.get(TextDoc, tid)
        if doc is not None and doc.status == "ready":
            enqueue(db, "ner", {"text_id": tid}, title=f"Имена собственные: «{doc.title}»")
            n += 1
    db.commit()
    flash(request, f"Извлечение имён поставлено в очередь для {n} текстов (прогресс — на странице «Задачи»)", "success")
    return redirect(back if back.startswith("/") else "/analysis?tab=entities")


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
    elif tab == "entities":
        ctx.update(_entities_context(db, request, dict_id, sel))
    elif tab == "geo":
        from app.services.geo import EDIT_KINDS, KIND_LABELS, place_map

        from app.services.dict_build import PLACE_MODES

        ctx.update({"geo": place_map(db, sel.ids), "GEO_KINDS": KIND_LABELS, "EDIT_KINDS": EDIT_KINDS, "PLACE_MODES": PLACE_MODES,
                    "geo_mode": "countries" if qp.get("geo_mode") == "countries" else "places"})
    elif tab == "semantic":
        ctx.update(_semantic_context(db, request, dict_id, sel, entry_id))
    elif tab == "topics":
        ctx.update(_topics_context(db, request, sel))
    elif tab == "stylo":
        ctx.update(_stylo_context(db, request, sel))
    elif tab == "textcmp":
        ctx.update(_textcmp_context(db, request, sel))
    elif tab == "freq":
        # отмеченные слова — только они (переключатель all_words показывает весь словарь)
        words = None if qp.get("all_words") else (sel.words or None)
        ctx["freq"] = A.only_words(A.frequency(db, dict_id, text_ids), words)
        ctx["heat"] = A.frequency_by_text(db, dict_id, text_ids, entry_ids=words, by="cat" if sel.unit == "cat" else "word")
        ctx["freq_filtered"] = bool(words)
        ctx["cats"] = None if ctx["no_cats"] else A.category_frequency(db, dict_id, text_ids, entry_ids=words, dim=qp.get("cat_dim") or "author")
        ctx["DIMENSIONS"] = A.DIMENSIONS
    elif tab == "conc":
        ctx["conc"] = A.concordance(
            db, dict_id, text_ids, entry_id=entry_id, entry_ids=sel.words, role=qp.get("role") or None,
            rel_type=qp.get("rel_type") or None, related_lemma=qp.get("related") or None,
            near=qp.get("near") or None, window=_int(qp.get("window")) or 3,
            phrase=qp.get("phrase") or None, phrase_any_order=bool(qp.get("phrase_any")),
            names_only=bool(qp.get("names")),
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
            # все связанные слова: первые PROFILE_TOP видны сразу, остальные — под «ещё …» (раньше редкие отбрасывались молча)
            ctx["profile"] = A.word_profile(db, entry_id, text_ids, scopes=scopes, include_negated=bool(qp.get("negated")),
                                            stop=sel.stopset, top=10_000)
            ctx["PROFILE_TOP"] = 25
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
    from app.services.unlink import is_unlinked, unlinked_pairs

    pairs = unlinked_pairs(db, [sentence_id]).get(sentence_id)
    unlinked = {r.related_idx for r in rels if is_unlinked(pairs, focus, r.related_idx)}
    rels.sort(key=lambda r: r.related_idx in unlinked)  # убранные вручную — в конце, с кнопкой «вернуть»
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
                  precise_pending=precise_pending, unlinked=unlinked,
                  tree_json=json.dumps(tree, ensure_ascii=False).replace("</", "<\\/"), groups=groups, hit_tokens=hit_tokens,
                  conllu=sent.to_conllu(sentence_id), prev_id=prev_id, next_id=next_id, context=context,
                  tree_children=tree_children)


@router.post("/sentences/{sentence_id}/unlink")
def sentence_unlink(request: Request, sentence_id: int, focus: int = Form(...), other: int = Form(...), undo: str = Form(""),
                    db: Session = Depends(get_session)):
    """Убрать связь двух слов предложения (или вернуть её — undo=1): ошибка разбора, например однородные сказуемые
    при чужом подлежащем. Учитывается в профиле слова, коллокациях и при пересчёте вхождений."""
    from app.services.unlink import relink, unlink

    row = db.get(SentenceRow, sentence_id)
    if row is None:
        raise HTTPException(404)
    a, b = sorted((focus, other))
    if a < 1 or b > (db.scalar(select(func.max(TokenRow.idx)).where(TokenRow.sentence_id == sentence_id)) or 0) or a == b:
        raise HTTPException(400)
    if undo:
        relink(db, sentence_id, focus, other)
        flash(request, "Связь возвращена.")
    else:
        n = unlink(db, sentence_id, focus, other)
        flash(request, "Связь убрана" + (f" (из сохранённых связей вхождений: {n})" if n else "") + ". Её можно вернуть.")
    return redirect(f"/sentences/{sentence_id}?token={focus}#rels")


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


# ------------------------------------------------------------------ проверка текста на признаки машинного текста


@router.get("/check")
def check_page(request: Request, text_id: str | None = None, db: Session = Depends(get_session)):
    from app.services import aicheck as C

    rep, label = None, ""
    tid = _int(text_id)
    if tid:
        doc = db.get(TextDoc, tid)
        if doc is not None and doc.status == "ready":
            sents = C.db_sentences(db, tid, limit_words=20000)
            rep, label = C.report(db, sents), f"«{doc.title}»" + (" (первые ~20 тыс. слов)" if (doc.n_words or 0) > 20000 else "")
    texts = list(db.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.source_type)
                            .where(TextDoc.status == "ready").order_by(TextDoc.title)))
    return render(request, "check.html", rep=rep, label=label, texts=texts, text_id=tid, pasted="", MIN_WORDS=C.MIN_WORDS)


@router.post("/check")
async def check_submit(request: Request, text: str = Form(""), file: UploadFile | None = File(None), db: Session = Depends(get_session)):
    from app.nlp.normalize import extract_text
    from app.services import aicheck as C

    body, label = text.strip(), "вставленный текст"
    if file is not None and file.filename:
        try:
            body = extract_text(file.filename, await file.read()).strip()
            label = f"файл «{file.filename}»"
        except Exception as exc:  # noqa: BLE001
            flash(request, f"Файл не прочитан: {exc}", "error")
    if not body:
        flash(request, "Вставьте текст или выберите файл", "error")
        return redirect("/check")
    words = body.split()
    if len(words) > 20000:
        body = " ".join(words[:20000])
        label += " (первые 20 тыс. слов)"
    rep = C.report(db, C.parse_raw(body))
    texts = list(db.execute(select(TextDoc.id, TextDoc.title, TextDoc.author, TextDoc.source_type)
                            .where(TextDoc.status == "ready").order_by(TextDoc.title)))
    return render(request, "check.html", rep=rep, label=label, texts=texts, text_id=None, pasted=text, MIN_WORDS=C.MIN_WORDS)


# ------------------------------------------------------------------ задачи


@router.get("/jobs")
def jobs_page(request: Request, db: Session = Depends(get_session)):
    from app.services.jobs import eta_seconds, human_duration, queue_estimate

    jobs = list(db.scalars(select(Job).order_by(Job.id.desc()).limit(100)))
    active = any(j.status in ("queued", "running") for j in jobs)
    q = queue_estimate(db) if active else {"n": 0, "seconds": None}
    return render(request, "jobs.html", jobs=jobs, active=active, etas={j.id: human_duration(eta_seconds(j)) for j in jobs},
                  queue_eta=human_duration(q["seconds"]), queue_n=q["n"], queue_unknown=q.get("unknown", 0))


@router.get("/api/jobs")
def jobs_api(db: Session = Depends(get_session)):
    from app.services.jobs import eta_seconds, human_duration, queue_estimate

    jobs = list(db.scalars(select(Job).where(Job.status.in_(["queued", "running"])).order_by(Job.id)))
    q = queue_estimate(db)
    return {"active": [{"id": j.id, "title": j.title, "status": j.status, "progress": j.progress, "message": j.message,
                        "eta": human_duration(eta_seconds(j))} for j in jobs],
            "queue": {"n": q["n"], "eta": human_duration(q["seconds"]), "unknown": q.get("unknown", 0)}}


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


# ------------------------------------------------------------------ база знаний


@router.get("/help")
def help_index(request: Request, q: str = ""):
    from app.services import help as H

    return render(request, "help.html", articles=list(H.articles().values()), terms=H.glossary(), q=q.strip(),
                  results=H.search(q) if q.strip() else None)


@router.get("/help/term/{tid}.json")
def help_term(tid: str):
    from app.services import help as H

    t = H.glossary().get(tid)
    if t is None:
        raise HTTPException(404)
    return JSONResponse(t)


@router.get("/help/files/{name}")
def help_file(name: str):
    from app.services import help as H

    p = H.FILES_DIR / name
    if "/" in name or "\\" in name or name.startswith(".") or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=name)


@router.get("/help/{slug}")
def help_article(request: Request, slug: str):
    from app.services import help as H

    art = H.articles().get(slug)
    if art is None:
        raise HTTPException(404)
    return render(request, "help_article.html", art=art, articles=list(H.articles().values()))


# ------------------------------------------------------------------ дашборд


def _dashboard_data(db: Session, request: Request) -> tuple[dict, object, int | None, list]:
    """Данные дашборда: в новом интерфейсе — дашборд v2 (services/dashboard_v2.py), в классическом — прежний."""
    from app.services.dashboard import build
    from app.services.dashboard_v2 import build_v2
    from app.services.selection import from_params

    sel = from_params(db, request.query_params)
    dictionaries = list(db.scalars(select(Dictionary).order_by(Dictionary.id)))
    dict_id = _int(request.query_params.get("dictionary_id")) or (dictionaries[0].id if dictionaries else None)
    return (build_v2 if ui_mode(request) == "v2" else build)(db, sel, dict_id), sel, dict_id, dictionaries


@router.get("/dashboard")
def dashboard_page(request: Request, db: Session = Depends(get_session)):
    """Обзорный дашборд по выборке: корпус (corpus_id), отмеченные тексты (texts) или один текст."""
    dash, sel, dict_id, dictionaries = _dashboard_data(db, request)
    return render(request, "dashboard_view.html", dash=dash, selq=sel.query(False), sel_pairs=sel.pairs(False),
                  dict_id=dict_id, dictionaries=dictionaries)

