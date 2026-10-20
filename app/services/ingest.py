"""Загрузка текста: нормализация → разбор → запись предложений и токенов (COPY)."""

from __future__ import annotations

import logging

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import PARSE_FAST, PARSE_PRECISE, Corpus, Dictionary, SentenceRow, TextDoc
from app.nlp.normalize import extract_text, normalize_text
from app.nlp.parsers import get_parser
from app.nlp.pipeline import count_segments, process_text
from app.nlp.syntax import SYNTAX_VERSION
from app.nlp.tree import format_feats, norm
from app.services.jobs import enqueue, set_progress

log = logging.getLogger(__name__)

SENT_COLS = (
    "id", "text_id", "idx", "paragraph_idx", "text", "n_words", "tree_depth", "n_clauses", "sent_type",
    "is_direct_speech", "has_homogeneous", "has_participial", "has_gerund", "parse_level",
)
TOKEN_COLS = (
    "sentence_id", "text_id", "idx", "form", "form_norm", "lemma", "lemma_norm", "upos", "feats", "head",
    "deprel", "start_char", "end_char",
)


def create_text(
    session: Session,
    *,
    filename: str,
    data: bytes,
    title: str | None = None,
    author: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    genre: str | None = None,
    source_type: str = "natural",
    generator: str | None = None,
    corpus_ids: list[int] | None = None,
    enqueue_job: bool = True,
    extra_meta: dict | None = None,
) -> TextDoc:
    raw = extract_text(filename, data)
    doc = TextDoc(
        title=(title or filename.rsplit(".", 1)[0]).strip()[:500],
        author=(author or None),
        year_from=year_from,
        year_to=year_to or year_from,
        genre=genre or None,
        source_type=source_type or "natural",
        generator=generator or None,
        source_filename=filename,
        raw_text=raw,
        norm_text=normalize_text(raw),
        status="queued",
        meta=dict(extra_meta or {}),
    )
    if corpus_ids:
        doc.corpora = list(session.scalars(select(Corpus).where(Corpus.id.in_(corpus_ids))))
    session.add(doc)
    session.flush()
    if enqueue_job:
        enqueue(session, "ingest", {"text_id": doc.id}, title=f"Разбор текста «{doc.title}»")
    return doc


def _alloc_ids(session: Session, seq: str, n: int) -> list[int]:
    rows = session.execute(text(f"SELECT nextval('{seq}') FROM generate_series(1, :n)"), {"n": n}).all()
    return [r[0] for r in rows]


def _copy(session: Session, table: str, cols: tuple[str, ...], rows: list[tuple]) -> None:
    raw = session.connection().connection.driver_connection
    with raw.cursor() as cur:
        with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)


def mark_interrupted(text_id: int, message: str) -> None:
    """Разбор не завершён (ошибка, отмена, падение обработчика): статус «ошибка», частичные предложения удаляются.

    Текст остаётся в базе — его можно разобрать заново кнопкой на странице текста.
    """
    with SessionLocal() as s:
        doc = s.get(TextDoc, text_id)
        if doc is None or doc.status == "ready":
            return
        s.execute(delete(SentenceRow).where(SentenceRow.text_id == text_id))
        doc.status, doc.error = "error", message
        doc.n_sentences = doc.n_tokens = doc.n_words = 0
        s.commit()


def pipeline_label() -> str:
    """Описание конвейера для texts.pipeline_version (от него зависит кэш статистического портрета)."""
    if settings.two_stage:
        return f"{get_parser(settings.fast_parser).version} → {get_parser().version} (двухступенчатый); syntax {SYNTAX_VERSION}"
    return f"{get_parser().version}; syntax {SYNTAX_VERSION}"


def run_ingest(job_id: int | None, text_id: int) -> None:
    # ступень 1: весь текст — быстрым парсером (или сразу основным, если двухступенчатая обработка выключена)
    if settings.two_stage:
        parser, level = get_parser(settings.fast_parser), PARSE_FAST
    else:
        parser, level = get_parser(), PARSE_PRECISE
    with SessionLocal() as s:
        doc = s.get(TextDoc, text_id)
        if doc is None:
            raise ValueError(f"Текст {text_id} не найден")
        s.execute(delete(SentenceRow).where(SentenceRow.text_id == text_id))
        doc.status = "processing"
        doc.error = None
        doc.meta = {k: v for k, v in (doc.meta or {}).items() if k not in ("stats", "ner")}  # статистика и имена пересчитаются
        s.commit()
        content = doc.norm_text or normalize_text(doc.raw_text)
        title = doc.title

    total = max(count_segments(content), 1)
    sent_idx = 0
    n_tokens = n_words = 0
    try:
        for batch in process_text(content, parser, settings.parse_batch_size):
            with SessionLocal() as s:
                ids = _alloc_ids(s, "sentences_id_seq", len(batch))
                srows, trows = [], []
                for sid, ps in zip(ids, batch):
                    sent_idx += 1
                    m = ps.metrics
                    srows.append((
                        sid, text_id, sent_idx, ps.paragraph_idx, ps.sentence.text, m["n_words"], m["tree_depth"],
                        m["n_clauses"], m["sent_type"], ps.is_direct_speech, m["has_homogeneous"],
                        m["has_participial"], m["has_gerund"], level,
                    ))
                    for t in ps.sentence.tokens:
                        trows.append((
                            sid, text_id, t.idx, t.form, norm(t.form), t.lemma, norm(t.lemma), t.upos,
                            format_feats(t.feats), t.head, t.deprel, t.start, t.end,
                        ))
                        n_tokens += 1
                        if t.upos != "PUNCT":
                            n_words += 1
                _copy(s, "sentences", SENT_COLS, srows)
                _copy(s, "tokens", TOKEN_COLS, trows)
                s.commit()
            if job_id:
                set_progress(job_id, sent_idx / total, f"Разобрано предложений: {sent_idx} из ~{total}")
    except Exception as exc:
        mark_interrupted(text_id, str(exc) or "Разбор прерван")
        raise

    with SessionLocal() as s:
        doc = s.get(TextDoc, text_id)
        doc.status = "ready"
        doc.n_sentences = sent_idx
        doc.n_tokens = n_tokens
        doc.n_words = n_words
        doc.pipeline_version = pipeline_label()
        # новые тексты сразу сопоставляем со всеми словарями (в двухступенчатом режиме здесь же
        # предложения со словами словарей получат точный разбор — ступень 2)
        for d in s.scalars(select(Dictionary)):
            enqueue(s, "match", {"dictionary_id": d.id, "text_ids": [text_id]},
                    title=f"Словарь «{d.name}» × текст «{title}»")
        if settings.ner_on_ingest:
            enqueue(s, "ner", {"text_id": text_id}, title=f"Имена собственные: «{title}»")
        s.commit()
    # статистический портрет считаем сразу, чтобы страница открывалась мгновенно
    try:
        from app.services.textstats import get_stats

        with SessionLocal() as s:
            get_stats(s, text_id, refresh=True)
    except Exception:  # noqa: BLE001 — статистика не должна ломать разбор
        log.exception("Не удалось посчитать статистику текста %s", text_id)
    log.info("Текст %s разобран: %s предложений, %s токенов", text_id, sent_idx, n_tokens)
