"""Пакетная загрузка текстов: много файлов (или ZIP-архив) + таблица метаданных.

Имя файла — ID текста в таблице (колонка «файл»): «1.txt», «001.txt» и «1» совпадают со строкой «1».
Порядок важности сведений: поле, заполненное вручную для файла, → строка таблицы → общие поля формы.
Прочерки «-» в таблице означают «нет данных». Колонка «корпус» (несколько — через «;») и флажок
«подкорпуса по авторам» раскладывают тексты по корпусам. Отчёт перечисляет файлы без строки в таблице
и строки таблицы без файла.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Corpus, TextDoc
from app.nlp.normalize import UnsupportedFormat
from app.services.ingest import create_text
from app.services.metadata import lookup, normalize_source_type, parse_meta_table, parse_years, split_corpora, unmatched_rows

TEXT_EXT = (".txt", ".docx", ".fb2")
META_EXT = (".csv", ".tsv", ".xlsx")
MAX_ZIP_TOTAL = 2 * 1024 ** 3  # защита от «zip-бомбы»: не больше 2 ГБ распакованного


@dataclass
class BatchReport:
    created: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    no_meta: list[str] = field(default_factory=list)
    rows_without_file: list[str] = field(default_factory=list)
    matched: int = 0
    corpora_created: list[str] = field(default_factory=list)
    meta_name: str = ""
    author_variants: list[tuple[str, ...]] = field(default_factory=list)


def _zip_name(info: zipfile.ZipInfo) -> str:
    """Имена в архивах Windows записаны в cp866 без флага UTF-8 — восстанавливаем кириллицу."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp866")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def expand(uploads: list[tuple[str, bytes]], max_mb: int, report: BatchReport) -> tuple[list[tuple[str, bytes]], tuple[str, bytes] | None]:
    """Файлы текстов (с распаковкой ZIP) и таблица метаданных, если она лежит внутри архива."""
    texts, meta = [], None
    for name, data in uploads:
        low = name.lower()
        if low.endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                report.errors.append(f"{name}: повреждённый ZIP-архив")
                continue
            total = sum(i.file_size for i in zf.infolist())
            if total > MAX_ZIP_TOTAL:
                report.errors.append(f"{name}: в архиве больше 2 ГБ данных")
                continue
            for info in zf.infolist():
                inner = _zip_name(info)
                base = PurePath(inner).name
                if info.is_dir() or base.startswith((".", "~$")) or "__MACOSX" in inner:
                    continue
                if base.lower().endswith(TEXT_EXT):
                    if info.file_size > max_mb * 1024 * 1024:
                        report.errors.append(f"{base}: файл больше {max_mb} МБ")
                        continue
                    texts.append((base, zf.read(info)))
                elif base.lower().endswith(META_EXT) and meta is None:
                    meta = (base, zf.read(info))
        elif low.endswith(META_EXT) and not low.endswith(".txt"):
            meta = meta or (name, data)
        else:
            if len(data) > max_mb * 1024 * 1024:
                report.errors.append(f"{name}: файл больше {max_mb} МБ")
                continue
            texts.append((name, data))
    return texts, meta


def _name_parts(author: str) -> tuple[str, str | None] | None:
    """(фамилия, первая буква имени): «Иван Сергеевич Шмелев» → («шмелев», «и»), «Шмелев И.» → («шмелев», «и»)."""
    if not author or "," in author:  # соавторы — не сравниваем
        return None
    words = [w for w in re.split(r"\s+", author.strip()) if w]
    full = [w for w in words if not w.endswith(".") and len(w) > 2]
    initials = [w[0].lower() for w in words if w.endswith(".") or len(w) <= 2]
    if not full:
        return None
    if len(full) == 1:  # «Шмелев И.» / «Горький»
        return full[0].lower().replace("ё", "е"), (initials[0] if initials else None)
    return full[-1].lower().replace("ё", "е"), full[0][0].lower()  # «Имя (Отчество) Фамилия»


def surname(author: str) -> str | None:
    p = _name_parts(author)
    return p[0] if p else None


def author_variants(authors: set[str], existing: set[str] = frozenset()) -> list[tuple[str, ...]]:
    """Разные написания, похожие на одного автора: та же фамилия и та же (или неизвестная) первая буква имени.
    «Иван Сергеевич Шмелев» и «Шмелев И.» — да; «Лев Николаевич Толстой» и «Алексей Николаевич Толстой» — нет."""
    pool = sorted(authors | set(existing))
    parts = {a: _name_parts(a) for a in pool}
    groups: list[set[str]] = []
    for i, a in enumerate(pool):
        for b in pool[i + 1 :]:
            pa, pb = parts[a], parts[b]
            if not pa or not pb or pa[0] != pb[0]:
                continue
            if pa[1] and pb[1] and pa[1] != pb[1]:
                continue
            for g in groups:
                if a in g or b in g:
                    g.update((a, b))
                    break
            else:
                groups.append({a, b})
    return [tuple(sorted(g)) for g in groups if g & authors]


def _corpus_id(session: Session, name: str, cache: dict, report: BatchReport) -> int:
    name = name.strip()[:255]
    if name in cache:
        return cache[name]
    c = session.scalar(select(Corpus).where(Corpus.name == name))
    if c is None:
        c = Corpus(name=name)
        session.add(c)
        session.flush()
        report.corpora_created.append(name)
    cache[name] = c.id
    return c.id


def run_batch(session: Session, uploads: list[tuple[str, bytes]], *, meta_upload: tuple[str, bytes] | None, common: dict,
              per_file: dict[str, dict], corpus_ids: list[int], by_author: bool, skip_existing: bool, max_mb: int) -> BatchReport:
    report = BatchReport()
    texts, zip_meta = expand(uploads, max_mb, report)
    meta: dict = {}
    source = meta_upload or zip_meta
    if source:
        report.meta_name = source[0]
        try:
            meta = parse_meta_table(source[0], source[1])
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"Таблица метаданных «{source[0]}» не прочитана: {exc}")
    existing = set()
    if skip_existing:
        existing = {(fn or "").lower() for fn in session.scalars(select(TextDoc.source_filename))}
    cache: dict[str, int] = {}
    used: set[int] = set()
    authors_seen: set[str] = set()
    for name, data in texts:
        m = lookup(meta, name) if meta else {}
        if m:
            used.add(id(m))
            report.matched += 1
        elif meta:
            report.no_meta.append(name)
        if skip_existing and name.lower() in existing:
            report.skipped.append(name)
            continue
        row = per_file.get(name, {})
        stem = PurePath(name).stem
        if row.get("title") == stem and m.get("title"):  # автозаполненное имя файла не должно перебивать таблицу
            row = {**row, "title": ""}

        def pick(key, default=""):
            return row.get(key) or m.get(key) or common.get(key, default) or default

        y1, y2 = parse_years(pick("years"))
        author = pick("author")
        authors_seen.add(author)
        cids = list(corpus_ids)
        for cname in split_corpora(m.get("corpus", "")):
            cids.append(_corpus_id(session, cname, cache, report))
        if by_author and author:
            cids.append(_corpus_id(session, author, cache, report))
        try:
            create_text(
                session, filename=name, data=data, title=pick("title") or None, author=author or None,
                year_from=y1, year_to=y2, genre=pick("genre") or None,
                source_type=normalize_source_type(row.get("source_type") or m.get("source_type")) or common.get("source_type") or "natural",
                generator=pick("generator") or None, corpus_ids=sorted(set(cids)),
                extra_meta={"file_id": m["_key"]} if m.get("_key") else None,
            )
            report.created += 1
        except UnsupportedFormat as exc:
            report.errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{name}: {exc}")
    if meta:
        report.rows_without_file = unmatched_rows(meta, used)
    batch_authors = {a for a in authors_seen if a}
    existing_authors = {a for a in session.scalars(select(TextDoc.author).distinct()) if a}
    report.author_variants = author_variants(batch_authors, existing_authors)
    return report
