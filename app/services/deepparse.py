"""Ступень 2 двухступенчатой обработки: точный разбор выбранных предложений основным парсером.

Токенизация не меняется (парсер получает те же токены, что и на ступени 1), поэтому строки tokens
обновляются на месте: id токенов сохраняются, ссылки на них (вхождения словарей) не ломаются.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import PARSE_FAST, PARSE_PRECISE, Dictionary, SentenceRow, TextDoc, TokenRow
from app.nlp.parsers import get_parser
from app.nlp.syntax import SentenceAnalysis
from app.nlp.tree import Sentence, Token, format_feats, norm
from app.services.jobs import enqueue, set_progress

log = logging.getLogger(__name__)


def fast_sentence_ids(session: Session, text_ids: list[int] | None = None, sentence_ids: list[int] | None = None) -> list[int]:
    q = select(SentenceRow.id).where(SentenceRow.parse_level == PARSE_FAST)
    if text_ids is not None:
        q = q.where(SentenceRow.text_id.in_(text_ids or [-1]))
    if sentence_ids is not None:
        q = q.where(SentenceRow.id.in_(sentence_ids or [-1]))
    return list(session.scalars(q.order_by(SentenceRow.id)))


def candidate_sentence_ids(session: Session, text_ids: list[int], forms: set[str], lemmas: set[str]) -> list[int]:
    """Предложения предварительного разбора, где может встретиться слово словаря.

    Отбор с запасом: любая форма слова (все омонимичные лексемы — «духи» и для «дух», и для «духи»)
    или совпадение предварительной леммы. Лишние кандидаты отсеет точный разбор и сопоставление.
    """
    if not text_ids or not (forms or lemmas):
        return []
    cond = []
    if forms:
        cond.append(TokenRow.form_norm.in_(sorted(forms)))
    if lemmas:
        cond.append(TokenRow.lemma_norm.in_(sorted(lemmas)))
    from sqlalchemy import or_

    q = (select(TokenRow.sentence_id).join(SentenceRow, SentenceRow.id == TokenRow.sentence_id)
         .where(TokenRow.text_id.in_(text_ids), SentenceRow.parse_level == PARSE_FAST, or_(*cond)).distinct())
    return sorted(session.scalars(q))


def upgrade_sentences(sentence_ids: list[int], job_id: int | None = None, progress: tuple[float, float] = (0.0, 1.0),
                      label: str = "Точный разбор предложений") -> int:
    """Переразбирает предложения основным парсером и помечает их PARSE_PRECISE. Возвращает число переразобранных."""
    if not sentence_ids:
        return 0
    parser = get_parser()
    batch = max(8, settings.parse_batch_size)
    done = 0
    lo, hi = progress
    for k in range(0, len(sentence_ids), batch):
        chunk = sentence_ids[k : k + batch]
        with SessionLocal() as s:
            # повторная проверка уровня: параллельная задача могла уже переразобрать эти предложения
            ids = fast_sentence_ids(s, sentence_ids=chunk)
            if ids:
                rows = s.execute(
                    select(TokenRow.id, TokenRow.sentence_id, TokenRow.idx, TokenRow.form, TokenRow.start_char, TokenRow.end_char)
                    .where(TokenRow.sentence_id.in_(ids)).order_by(TokenRow.sentence_id, TokenRow.idx)
                ).all()
                by_sent: dict[int, list] = {}
                for r in rows:
                    by_sent.setdefault(r.sentence_id, []).append(r)
                texts = dict(s.execute(select(SentenceRow.id, SentenceRow.text).where(SentenceRow.id.in_(ids))).all())
                order = [sid for sid in ids if by_sent.get(sid)]
                parsed = parser.parse([[t.form for t in by_sent[sid]] for sid in order])
                tok_updates, sent_updates = [], []
                for sid, words in zip(order, parsed):
                    toks = by_sent[sid]
                    if len(words) != len(toks):  # не должно случаться: токены переданы готовыми
                        log.warning("Предложение %s: парсер вернул %s токенов вместо %s — пропущено", sid, len(words), len(toks))
                        continue
                    tokens = [Token(t.idx, t.form, w.lemma, w.upos, w.feats, w.head, w.deprel, t.start_char, t.end_char)
                              for t, w in zip(toks, words)]
                    m = SentenceAnalysis(Sentence(tokens, texts.get(sid, ""))).metrics()
                    sent_updates.append((sid, m["tree_depth"], m["n_clauses"], m["sent_type"], m["has_homogeneous"],
                                         m["has_participial"], m["has_gerund"]))
                    for t, w in zip(toks, words):
                        tok_updates.append((t.id, w.lemma, norm(w.lemma), w.upos, format_feats(w.feats), w.head, w.deprel))
                _apply(s, tok_updates, sent_updates)
                from app.services.parse_edit import reapply

                reapply(s, sentence_ids=[u[0] for u in sent_updates])  # ручные правки разбора — поверх нового разбора
                s.commit()
                done += len(sent_updates)
        if job_id:
            frac = min(1.0, (k + len(chunk)) / len(sentence_ids))
            set_progress(job_id, lo + (hi - lo) * frac, f"{label}: {min(k + len(chunk), len(sentence_ids))} из {len(sentence_ids)}")
    return done


def _apply(session: Session, tok_updates: list[tuple], sent_updates: list[tuple]) -> None:
    """Массовое обновление через временные таблицы и COPY (быстро даже для тысяч предложений)."""
    raw = session.connection().connection.driver_connection
    with raw.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tok_upd (id bigint, lemma text, lemma_norm text, upos text, feats text, "
                    "head smallint, deprel text) ON COMMIT DELETE ROWS")
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _sent_upd (id bigint, tree_depth smallint, n_clauses smallint, "
                    "sent_type text, has_homogeneous boolean, has_participial boolean, has_gerund boolean) ON COMMIT DELETE ROWS")
        with cur.copy("COPY _tok_upd FROM STDIN") as cp:
            for r in tok_updates:
                cp.write_row(r)
        with cur.copy("COPY _sent_upd FROM STDIN") as cp:
            for r in sent_updates:
                cp.write_row(r)
    session.execute(text(
        "UPDATE tokens t SET lemma = u.lemma, lemma_norm = u.lemma_norm, upos = u.upos, feats = u.feats, "
        "head = u.head, deprel = u.deprel FROM _tok_upd u WHERE t.id = u.id"))
    session.execute(text(
        "UPDATE sentences s SET tree_depth = u.tree_depth, n_clauses = u.n_clauses, sent_type = u.sent_type, "
        "has_homogeneous = u.has_homogeneous, has_participial = u.has_participial, has_gerund = u.has_gerund, "
        f"parse_level = {PARSE_PRECISE} FROM _sent_upd u WHERE s.id = u.id"))


def run_deepparse(job_id: int | None, text_id: int | None = None, sentence_ids: list[int] | None = None) -> None:
    """Задача «Точный разбор»: всего текста (text_id) или отдельных предложений.

    После переразбора словари пересопоставляются с затронутыми текстами: точный разбор мог найти
    вхождения, которые не попали в кандидаты (другая лемма у предварительного разбора).
    """
    with SessionLocal() as s:
        if text_id is not None:
            ids = fast_sentence_ids(s, text_ids=[text_id])
        else:
            ids = fast_sentence_ids(s, sentence_ids=sentence_ids or [])
        text_ids = sorted(set(s.scalars(select(SentenceRow.text_id).where(SentenceRow.id.in_(ids or [-1])))))
    n = upgrade_sentences(ids, job_id, (0.0, 1.0))
    log.info("Точный разбор: %s предложений", n)
    if text_id is not None or text_ids:
        with SessionLocal() as s:
            targets = [text_id] if text_id is not None else text_ids
            titles = dict(s.execute(select(TextDoc.id, TextDoc.title).where(TextDoc.id.in_(targets))).all())
            for d in s.scalars(select(Dictionary)):
                enqueue(s, "match", {"dictionary_id": d.id, "text_ids": targets},
                        title=f"Словарь «{d.name}» × {', '.join('«' + t + '»' for t in titles.values())[:200]}")
            s.commit()
        if text_id is not None:  # портрет пересчитывается: синтаксис теперь точнее
            try:
                from app.services.textstats import get_stats

                with SessionLocal() as s:
                    get_stats(s, text_id, refresh=True)
            except Exception:  # noqa: BLE001
                log.exception("Не удалось пересчитать портрет текста %s", text_id)


def parse_levels(session: Session, text_id: int) -> dict:
    """Сколько предложений текста разобрано точно и предварительно — для карточки текста и портрета."""
    rows = dict(session.execute(
        select(SentenceRow.parse_level, text("count(*)")).where(SentenceRow.text_id == text_id).group_by(SentenceRow.parse_level)
    ).all())
    precise, fast = rows.get(PARSE_PRECISE, 0), rows.get(PARSE_FAST, 0)
    return {"precise": precise, "fast": fast, "total": precise + fast}
