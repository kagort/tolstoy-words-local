"""Выгрузка корпуса архивом ZIP: очищенные тексты, таблица метаданных, по желанию — леммы и разметка CoNLL-U.

Очищенный текст — тот, что разбирался (нормализация: склеенные переносы, лишние пробелы, единые кавычки и тире —
nlp/normalize.py), абзацы через пустую строку. Леммы — по предложению на строку, без знаков препинания. Архив
собирается во временном файле на диске (большой корпус не держится в памяти целиком).
"""

from __future__ import annotations

import csv
import io
import re
import tempfile
import zipfile

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Corpus, SentenceRow, TextDoc, TokenRow, corpus_texts
from app.nlp.normalize import normalize_text, paragraphs
from app.services.labels import short_author

PARTS = {"clean": "очищенные тексты (.txt)", "lemmas": "леммы (.txt, предложение на строку)", "conllu": "разметка CoNLL-U"}
SOURCE_TYPES = {"natural": "естественный", "generated": "сгенерированный", "other": "другое"}


_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f",
                "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def translit(s: str) -> str:
    """Латиницей — для имени скачиваемого файла (русское имя в заголовке браузеры понимают не всегда)."""
    return "".join(_TR.get(ch, _TR.get(ch.lower(), ch).capitalize() if ch.lower() in _TR else ch) for ch in s)


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", (s or "").strip(), flags=re.UNICODE).strip("_")
    return s[:n] or "text"


def file_stem(doc: TextDoc) -> str:
    """«012_Булгаков_Собачье_сердце» — номер для порядка и уникальности, автор и название — чтобы понять без таблицы."""
    author = short_author(doc.author).replace(".", "").replace(" ", "_") if doc.author else ""
    return f"{doc.id:04d}_" + "_".join(x for x in (_slug(author, 30), _slug(doc.title)) if x)


def corpus_docs(session: Session, corpus_id: int | None) -> tuple[str, list[TextDoc]]:
    """Готовые тексты корпуса (None — все тексты базы) и название для имени архива."""
    q = select(TextDoc).where(TextDoc.status == "ready").order_by(TextDoc.id)
    name = "все тексты"
    if corpus_id is not None:
        c = session.get(Corpus, corpus_id)
        if c is None:
            return "", []
        name = c.name
        q = q.where(TextDoc.id.in_(select(corpus_texts.c.text_id).where(corpus_texts.c.corpus_id == corpus_id)))
    return name, list(session.scalars(q))


def _lemma_lines(session: Session, text_id: int):
    sid_prev, words = None, []
    q = (select(TokenRow.sentence_id, TokenRow.lemma).join(SentenceRow, SentenceRow.id == TokenRow.sentence_id)
         .where(TokenRow.text_id == text_id, TokenRow.upos != "PUNCT").order_by(SentenceRow.idx, TokenRow.idx))
    for sid, lemma in session.execute(q):
        if sid != sid_prev and words:
            yield " ".join(words)
            words = []
        sid_prev = sid
        words.append(lemma)
    if words:
        yield " ".join(words)


def _conllu(session: Session, doc: TextDoc):
    from app.services.matching import load_sentences

    yield f"# newdoc id = {doc.id}\n# title = {doc.title}\n# author = {doc.author or ''}\n# pipeline = {doc.pipeline_version}\n\n"
    sids = list(session.scalars(select(SentenceRow.id).where(SentenceRow.text_id == doc.id).order_by(SentenceRow.idx)))
    for k in range(0, len(sids), 500):
        sents = load_sentences(session, sids[k: k + 500])
        for sid in sids[k: k + 500]:
            if sid in sents:
                yield sents[sid].to_conllu(sid) + "\n"


def build_zip(session: Session, corpus_id: int | None, parts: set[str]) -> tuple[str, tempfile.SpooledTemporaryFile, int]:
    """(название, файл архива с позицией 0, число текстов). parts ⊆ PARTS; «clean» и таблица метаданных — всегда."""
    name, docs = corpus_docs(session, corpus_id)
    parts = (set(parts) & set(PARTS)) | {"clean"}
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        meta = io.StringIO()
        w = csv.writer(meta, delimiter=";")
        w.writerow(["ID", "файл", "название", "автор", "год", "год (по)", "жанр", "тип", "модель/промпт", "слов",
                    "предложений", "исходный файл", "конвейер разбора"])
        for doc in docs:
            stem = file_stem(doc)
            clean = doc.norm_text or normalize_text(doc.raw_text or "")
            z.writestr(f"texts/{stem}.txt", "\n\n".join(paragraphs(clean)).rstrip() + "\n")  # абзацы — через пустую строку
            if "lemmas" in parts:
                z.writestr(f"lemmas/{stem}.txt", "\n".join(_lemma_lines(session, doc.id)) + "\n")
            if "conllu" in parts:
                z.writestr(f"conllu/{stem}.conllu", "".join(_conllu(session, doc)))
            w.writerow([doc.id, f"{stem}.txt", doc.title, doc.author or "", doc.year_from or "", doc.year_to or "",
                        doc.genre or "", SOURCE_TYPES.get(doc.source_type, doc.source_type or ""), doc.generator or "",
                        doc.n_words, doc.n_sentences, doc.source_filename or "", doc.pipeline_version or ""])
        z.writestr("metadata.csv", "﻿" + meta.getvalue())  # BOM — Excel открывает кириллицу без настройки
        z.writestr("README.txt", _readme(name, len(docs), parts))
    buf.seek(0)
    return name, buf, len(docs)


def _readme(name: str, n: int, parts: set[str]) -> str:
    lines = [f"Корпус «{name}»: {n} текстов (выгрузка Quotverba).", "",
             "texts/     — очищенные тексты: нормализованы (склеены переносы, единые кавычки и тире, лишние пробелы",
             "             убраны), абзацы через пустую строку; кодировка UTF-8.",
             "metadata.csv — таблица метаданных (разделитель «;», UTF-8): ID, файл, название, автор, год, жанр, тип…"]
    if "lemmas" in parts:
        lines.append("lemmas/    — леммы: предложение на строку, слова через пробел, без знаков препинания.")
    if "conllu" in parts:
        lines.append("conllu/    — морфологическая и синтаксическая разметка в формате CoNLL-U (Universal Dependencies).")
    return "\n".join(lines) + "\n"
