"""Сопоставление словаря с корпусом: вхождения, роли и связи токенов."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import and_, delete, not_, or_, select, update
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import (
    PARSE_PRECISE,
    Dictionary,
    DictionaryEntry,
    Hit,
    Job,
    SentenceRow,
    TextDoc,
    TokenRow,
)
from app.nlp.morph import word_forms
from app.nlp.syntax import SYNTAX_VERSION, SentenceAnalysis
from app.nlp.tree import Sentence, Token, parse_feats
from app.services.ingest import _alloc_ids, _copy
from app.services.jobs import set_progress
from app.services.unlink import is_unlinked, unlinked_pairs

log = logging.getLogger(__name__)

HIT_COLS = (
    "id", "entry_id", "dictionary_id", "token_id", "sentence_id", "text_id", "token_idx", "form", "role",
    "role_subtype", "role_detail", "in_coord_group", "syntax_version",
)
REL_COLS = (
    "hit_id", "dictionary_id", "entry_id", "text_id", "rel_type", "direction", "related_idx", "related_lemma",
    "related_upos", "detail", "path", "source", "scope", "rule", "confidence", "negated",
)

CHUNK = 500


def relation_row(hit_id: int, dictionary_id: int, entry_id: int, text_id: int, rel) -> tuple:
    """Строка token_relations (порядок REL_COLS) для связи вхождения."""
    return (hit_id, dictionary_id, entry_id, text_id, rel.rel_type, rel.direction, rel.related_idx, rel.related_lemma,
            rel.related_upos, rel.detail or None, rel.path or None, rel.source, rel.scope, rel.rule or None,
            rel.confidence, rel.negated)


def entry_token_filter(entry: DictionaryEntry, settings: dict):
    """Условие SQL на таблицу tokens для слова словаря."""
    conds = []
    if entry.match_mode == "forms":
        conds.append(TokenRow.form_norm.in_(sorted(word_forms(entry.lemma, entry.upos))))
    else:
        conds.append(TokenRow.lemma_norm == entry.lemma_norm)
    if entry.upos and entry.upos != "X":
        allowed = {entry.upos}
        if entry.upos == "NOUN" and settings.get("merge_propn", True):
            allowed.add("PROPN")
        if entry.upos == "VERB":
            allowed.add("AUX")
        if entry.upos == "PROPN":  # имя: парсер мог разобрать как существительное — берём, если с заглавной («Вера», не «вера»)
            conds.append(or_(TokenRow.upos == "PROPN", and_(TokenRow.upos == "NOUN", TokenRow.form.op("~")("^[А-ЯЁA-Z]"))))
        else:
            conds.append(TokenRow.upos.in_(sorted(allowed)))
        if entry.upos in ("VERB", "ADJ"):
            if not settings.get("include_participles", False):
                conds.append(not_(TokenRow.feats.like("%VerbForm=Part%")))
            if not settings.get("include_gerunds", False):
                conds.append(not_(TokenRow.feats.like("%VerbForm=Conv%")))
    return and_(*conds)


def load_sentences(session: Session, sentence_ids: list[int]) -> dict[int, Sentence]:
    rows = session.execute(
        select(TokenRow.sentence_id, TokenRow.idx, TokenRow.form, TokenRow.lemma, TokenRow.upos, TokenRow.feats,
               TokenRow.head, TokenRow.deprel, TokenRow.start_char, TokenRow.end_char)
        .where(TokenRow.sentence_id.in_(sentence_ids))
        .order_by(TokenRow.sentence_id, TokenRow.idx)
    ).all()
    by_sent: dict[int, list[Token]] = defaultdict(list)
    for sid, idx, form, lemma, upos, feats, head, deprel, st, en in rows:
        by_sent[sid].append(Token(idx, form, lemma, upos, parse_feats(feats), head, deprel, st, en))
    texts = dict(session.execute(select(SentenceRow.id, SentenceRow.text).where(SentenceRow.id.in_(sentence_ids))).all())
    return {sid: Sentence(toks, texts.get(sid, "")) for sid, toks in by_sent.items()}


def run_match(job_id: int | None, dictionary_id: int, entry_ids: list[int] | None = None,
              text_ids: list[int] | None = None, manual: list | None = None) -> None:
    with SessionLocal() as s:
        d = s.get(Dictionary, dictionary_id)
        if d is None:
            return
        settings = dict(d.settings or {})
        entries = [e for e in d.entries if not entry_ids or e.id in entry_ids]
        ready_texts = list(s.scalars(select(TextDoc.id).where(TextDoc.status == "ready")))
        if text_ids:
            ready_texts = [t for t in ready_texts if t in set(text_ids)]
        from app.services.dictionaries import scope_text_ids

        scope = scope_text_ids(s, d)  # словарь только для выбранных корпусов
        if scope is not None:
            ready_texts = [t for t in ready_texts if t in scope]
        # удаляем старые результаты для пересчитываемой области
        q = delete(Hit).where(Hit.dictionary_id == dictionary_id)
        if entry_ids:
            q = q.where(Hit.entry_id.in_(entry_ids))
        if text_ids:
            q = q.where(Hit.text_id.in_(text_ids))
        # ручные отметки «ложное вхождение» переживают пересчёт
        keep_q = select(Hit.entry_id, Hit.token_id, Hit.is_valid).where(
            Hit.dictionary_id == dictionary_id, Hit.is_valid.is_not(None))
        if entry_ids:
            keep_q = keep_q.where(Hit.entry_id.in_(entry_ids))
        if text_ids:
            keep_q = keep_q.where(Hit.text_id.in_(text_ids))
        # отметки сохраняются в параметрах задачи в той же транзакции, что и удаление, — поэтому они
        # не теряются, даже если обработчик упадёт посреди пересчёта и задача запустится заново
        marks = {(e, t): v for e, t, v in (manual or [])}
        marks.update({(e, t): v for e, t, v in s.execute(keep_q).all()})
        manual = [[e, t, v] for (e, t), v in marks.items()]
        if job_id:
            job = s.get(Job, job_id)
            job.params = {**(job.params or {}), "manual": manual}
        s.execute(q)
        s.commit()
        entry_data = [(e.id, e.surface, entry_token_filter(e, settings), e.phrase) for e in entries]
        # двухступенчатая обработка: слова словаря ищутся с запасом в предварительном разборе
        cand_forms, cand_lemmas = set(), set()
        for e in entries:
            cand_forms |= word_forms(e.lemma, e.upos if e.match_mode == "forms" else None)
            cand_lemmas.add(e.lemma_norm)
        from app.services.deepparse import candidate_sentence_ids

        candidates = candidate_sentence_ids(s, ready_texts, cand_forms, cand_lemmas) if entries else []

    total_hits = 0
    try:
        # ступень 2: кандидаты получают точный разбор основным парсером, только потом ищутся вхождения
        match_from = 0.0
        if candidates:
            from app.services.deepparse import upgrade_sentences

            match_from = 0.7
            upgrade_sentences(candidates, job_id, (0.0, match_from), label="Точный разбор предложений со словами словаря")
        total_hits = _match_entries(job_id, dictionary_id, entry_data, ready_texts, match_from)
    finally:  # и при отмене задачи ручные отметки возвращаются на уже найденные вхождения
        with SessionLocal() as s:
            for e_id, tok_id, valid in manual:
                s.execute(update(Hit).where(Hit.entry_id == e_id, Hit.token_id == tok_id).values(is_valid=valid))
            s.commit()
    with SessionLocal() as s:
        d = s.get(Dictionary, dictionary_id)
        if d is not None:
            d.matched_at = datetime.now(timezone.utc)
            if not entry_ids:  # какие тексты посчитаны: полный пересчёт — заново, по текстам (корпусу) — добавить к прежним
                st = dict(d.settings or {})
                done = set(ready_texts) | (set(st.get("matched_texts") or []) if text_ids else set())
                d.settings = {**st, "matched_texts": sorted(done)}
            flag_name_hits(s, [dictionary_id], text_ids)  # слова внутри имён собственных не считаются
            s.commit()
    log.info("Словарь %s: найдено вхождений %s", dictionary_id, total_hits)


def _match_entries(job_id: int | None, dictionary_id: int, entry_data: list, ready_texts: list[int],
                   progress_from: float = 0.0) -> int:
    """Вхождения ищутся только в предложениях точного разбора (PARSE_PRECISE)."""
    total_hits = 0
    span = 1.0 - progress_from
    precise = select(SentenceRow.id).where(SentenceRow.text_id.in_(ready_texts or [-1]), SentenceRow.parse_level == PARSE_PRECISE)
    from app.services.phrases import contains_head

    for n, (entry_id, surface, cond, phrase) in enumerate(entry_data, start=1):
        if not ready_texts:
            break
        with SessionLocal() as s:
            rows = s.execute(
                select(TokenRow.id, TokenRow.sentence_id, TokenRow.text_id, TokenRow.idx, TokenRow.form)
                .where(cond, TokenRow.text_id.in_(ready_texts), TokenRow.sentence_id.in_(precise))
                .order_by(TokenRow.sentence_id, TokenRow.idx)
            ).all()
            for start in range(0, len(rows), CHUNK):
                chunk = rows[start : start + CHUNK]
                sents = load_sentences(s, sorted({r.sentence_id for r in chunk}))
                unlinked = unlinked_pairs(s, sents)  # связи, убранные вручную на странице предложения
                analyses: dict[int, SentenceAnalysis] = {}
                hit_ids = _alloc_ids(s, "dictionary_hits_id_seq", len(chunk))
                hrows, rrows = [], []
                for hid, r in zip(hit_ids, chunk):
                    sent = sents.get(r.sentence_id)
                    if sent is None:
                        continue
                    # словосочетание: главное слово засчитывается, только если рядом есть остальные слова
                    if phrase and not contains_head(sent.tokens, phrase["slots"], phrase["head"], r.idx):
                        continue
                    an = analyses.get(r.sentence_id) or analyses.setdefault(r.sentence_id, SentenceAnalysis(sent))
                    role = an.role(r.idx)
                    hrows.append((hid, entry_id, dictionary_id, r.id, r.sentence_id, r.text_id, r.idx, r.form,
                                  role.role, role.subtype[:64] or None, role.detail[:64] or None,
                                  an.group_of(r.idx) is not None, SYNTAX_VERSION))
                    for rel in an.relations(r.idx):
                        if not is_unlinked(unlinked.get(r.sentence_id), r.idx, rel.related_idx):
                            rrows.append(relation_row(hid, dictionary_id, entry_id, r.text_id, rel))
                _copy(s, "dictionary_hits", HIT_COLS, hrows)
                _copy(s, "token_relations", REL_COLS, rrows)
                s.commit()
                total_hits += len(hrows)
                if job_id and start + CHUNK < len(rows):  # пульс и проверка отмены внутри частого слова
                    set_progress(job_id, progress_from + span * (n - 1 + (start + CHUNK) / len(rows)) / max(len(entry_data), 1),
                                 f"Обработано слов: {n - 1} из {len(entry_data)} («{surface}»: {start + CHUNK} из {len(rows)})")
        if job_id:
            set_progress(job_id, progress_from + span * n / max(len(entry_data), 1), f"Обработано слов: {n} из {len(entry_data)} («{surface}»)")
    return total_hits


def flag_name_hits(session: Session, dictionary_ids: list[int] | None = None, text_ids: list[int] | None = None) -> int:
    """Помечает вхождения слов словаря, стоящие внутри имени персоны (вкладка «Имена»): «Любовь» в «Любовь
    Андреевна» для слова «любовь», «Вера» (Вера Ростова) для слова «вера». Такие вхождения не считаются
    (analytics.VALID), пока их не отметят ✓ вручную. Места и организации («Америка», «НАТО») и записи словаря с частью
    речи PROPN не помечаются — их в словарь добавляют намеренно. Настройки словаря: count_names — считать и внутри
    имён; names_ok_entries — слова-исключения.

    Вызывается после сопоставления, после извлечения имён и после правки имён. Возвращает число помеченных вхождений.
    """
    from sqlalchemy import text as sql

    dicts = list(session.scalars(select(Dictionary).where(Dictionary.id.in_(dictionary_ids)) if dictionary_ids
                                 else select(Dictionary)))
    flagged = 0
    for d in dicts:
        st = d.settings or {}
        params = {"d": d.id, "ok": [int(x) for x in st.get("names_ok_entries", [])] or [-1],
                  "tids": text_ids or [-1], "all_texts": not text_ids}
        scope = "h.dictionary_id = :d AND (:all_texts OR h.text_id = ANY(:tids))"
        if st.get("count_names"):
            session.execute(sql(f"UPDATE dictionary_hits h SET in_name = false WHERE {scope} AND h.in_name"), params)
            continue
        session.execute(sql(f"""
            UPDATE dictionary_hits h SET in_name = (h.entry_id <> ALL(:ok) AND EXISTS (
                SELECT 1 FROM entities e
                JOIN tokens t ON t.id = h.token_id
                JOIN dictionary_entries de ON de.id = h.entry_id
                LEFT JOIN entity_overrides o ON o.norm = e.norm
                WHERE e.sentence_id = h.sentence_id AND t.start_char >= e.start_char AND t.end_char <= e.end_char
                  AND coalesce(o.hidden, false) = false
                  AND coalesce(o.etype, e.etype) = 'PER'
                  AND coalesce(de.upos, '') <> 'PROPN'))
            WHERE {scope}"""), params)
        flagged += session.scalar(sql(f"SELECT count(*) FROM dictionary_hits h WHERE {scope} AND h.in_name"), params) or 0
    return flagged
