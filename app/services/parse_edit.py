"""Ручная правка разбора на странице предложения: часть речи, главное слово и тип связи слова (модель TokenEdit).

После правки предложение пересчитывается для всех словарей (роли, связи, вхождения — ручные отметки ✓/✗
сохраняются), метрики предложения обновляются. Правка переживает точный разбор и повторный разбор текста:
deepparse.upgrade_sentences и ingest.run_ingest вызывают reapply.
"""

from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models import PARSE_PRECISE, Dictionary, Hit, SentenceRow, TokenEdit, TokenRelation, TokenRow
from app.nlp.morph import POS_LABELS, word_forms
from app.nlp.syntax import SYNTAX_VERSION, SentenceAnalysis
from app.nlp.tree import norm

# типы связи UD для формы правки: основные (без подтипов — их можно ввести вручную, «nsubj:pass»)
UD_RELS = {
    "root": "корень (сказуемое главного предложения)",
    "nsubj": "подлежащее",
    "nsubj:pass": "подлежащее при страдательном",
    "csubj": "подлежащее-придаточное",
    "obj": "прямое дополнение",
    "iobj": "косвенное дополнение",
    "obl": "обстоятельство / дополнение с предлогом",
    "obl:agent": "дополнение-деятель («кем»)",
    "amod": "определение (прилагательное)",
    "nmod": "несогласованное определение",
    "det": "определитель (мой, этот)",
    "nummod": "числительное при существительном",
    "appos": "приложение",
    "acl": "причастный оборот",
    "acl:relcl": "придаточное определительное",
    "advmod": "обстоятельство (наречие)",
    "advcl": "обстоятельственное придаточное / деепричастие",
    "xcomp": "часть сказуемого (инфинитив)",
    "ccomp": "придаточное изъяснительное",
    "conj": "однородный член",
    "cc": "сочинительный союз",
    "case": "предлог",
    "mark": "подчинительный союз",
    "cop": "связка",
    "aux": "вспомогательный глагол",
    "aux:pass": "вспомогательный (страдательный)",
    "parataxis": "вставка, бессоюзная связь",
    "flat:name": "часть имени",
    "flat": "неделимое сочетание",
    "fixed": "составной служебный элемент",
    "compound": "сложное слово",
    "discourse": "частица, междометие",
    "vocative": "обращение",
    "orphan": "при пропущенном слове",
    "expl": "формальное слово",
    "dep": "неопределённая связь",
    "punct": "знак препинания",
}
UD_BASE = {r.split(":")[0] for r in UD_RELS}


def _sentence_edits(session: Session, row: SentenceRow) -> dict[int, TokenEdit]:
    return {e.idx: e for e in session.scalars(select(TokenEdit).where(TokenEdit.text_id == row.text_id, TokenEdit.sent_idx == row.idx))}


def edits_for(session: Session, sentence_id: int) -> dict[int, TokenEdit]:
    row = session.get(SentenceRow, sentence_id)
    return _sentence_edits(session, row) if row else {}


def validate(tokens: list, idx: int, upos: str, head: int, deprel: str) -> str | None:
    """Ошибка правки (текстом) или None. tokens — строки предложения с idx, head."""
    n = len(tokens)
    if upos not in POS_LABELS:
        return f"неизвестная часть речи «{upos}»"
    if not deprel or deprel.split(":")[0] not in UD_BASE:
        return f"неизвестный тип связи «{deprel}»"
    if head == idx:
        return "слово не может зависеть от самого себя"
    if not 0 <= head <= n:
        return "нет такого главного слова"
    if (head == 0) != (deprel == "root"):
        return "у корня (главное слово «— корень —») тип связи root, и наоборот"
    heads = {t.idx: t.head for t in tokens}
    heads[idx] = head
    seen, cur = set(), head
    while cur:  # от нового главного слова вверх к корню: если встретим само слово — получится цикл
        if cur == idx:
            return "получается цикл: главное слово само зависит от этого слова"
        if cur in seen:
            break
        seen.add(cur)
        cur = heads.get(cur, 0)
    return None


def save_edit(session: Session, sentence_id: int, idx: int, upos: str, head: int, deprel: str) -> dict:
    """Записать правку и пересчитать предложение. Возвращает {"error": …} или {"changed": …, "hits": …}."""
    row = session.get(SentenceRow, sentence_id)
    toks = list(session.scalars(select(TokenRow).where(TokenRow.sentence_id == sentence_id).order_by(TokenRow.idx)))
    tok = next((t for t in toks if t.idx == idx), None)
    if row is None or tok is None:
        return {"error": "нет такого слова"}
    deprel = deprel.strip()
    err = validate(toks, idx, upos, head, deprel)
    if err:
        return {"error": err}
    if (tok.upos, tok.head, tok.deprel) == (upos, head, deprel):
        return {"changed": False}
    edit = _sentence_edits(session, row).get(idx)
    if edit is None:
        edit = TokenEdit(text_id=row.text_id, sent_idx=row.idx, idx=idx, form=tok.form,
                         orig_upos=tok.upos, orig_head=tok.head, orig_deprel=tok.deprel)
        session.add(edit)
    edit.upos, edit.head, edit.deprel = upos, head, deprel
    if (edit.orig_upos, edit.orig_head, edit.orig_deprel) == (upos, head, deprel):  # вернули как было
        session.delete(edit)
    tok.upos, tok.head, tok.deprel = upos, head, deprel
    session.flush()
    return {"changed": True, "hits": refresh_sentence(session, sentence_id)}


def revert(session: Session, sentence_id: int, idx: int | None = None) -> int:
    """Вернуть разбор парсера для слова (или для всех слов предложения); возвращает число снятых правок."""
    row = session.get(SentenceRow, sentence_id)
    if row is None:
        return 0
    edits = [e for i, e in _sentence_edits(session, row).items() if idx is None or i == idx]
    for e in edits:
        session.execute(update(TokenRow).where(TokenRow.sentence_id == sentence_id, TokenRow.idx == e.idx)
                        .values(upos=e.orig_upos, head=e.orig_head, deprel=e.orig_deprel))
        session.delete(e)
    if edits:
        session.flush()
        refresh_sentence(session, sentence_id)
    return len(edits)


def reapply(session: Session, sentence_ids: list[int] | None = None, text_id: int | None = None) -> int:
    """Наложить сохранённые правки на свежий разбор (после точного разбора или повторного разбора текста).

    Новый разбор парсера становится «как было» (orig_*). Слово, которого нет на месте (другая форма), пропускается —
    правка остаётся в базе. Вхождения не пересчитываются: это делает следующая задача сопоставления.
    """
    q = select(SentenceRow.id, SentenceRow.text_id, SentenceRow.idx)
    if sentence_ids is not None:
        q = q.where(SentenceRow.id.in_(sentence_ids or [-1]))
    if text_id is not None:
        q = q.where(SentenceRow.text_id == text_id)
    texts = {text_id} if text_id is not None else None
    if texts is None:
        rows = session.execute(q).all()
        texts = {r.text_id for r in rows}
    if not texts:
        return 0
    edits = session.scalars(select(TokenEdit).where(TokenEdit.text_id.in_(texts))).all()
    if not edits:
        return 0
    by_pos = {(e.text_id, e.sent_idx): [] for e in edits}
    for e in edits:
        by_pos[(e.text_id, e.sent_idx)].append(e)
    applied, touched = 0, set()
    for sid, tid, sidx in session.execute(q):
        for e in by_pos.get((tid, sidx), []):
            tok = session.scalar(select(TokenRow).where(TokenRow.sentence_id == sid, TokenRow.idx == e.idx))
            if tok is None or tok.form != e.form:
                continue
            e.orig_upos, e.orig_head, e.orig_deprel = tok.upos, tok.head, tok.deprel
            tok.upos, tok.head, tok.deprel = e.upos, e.head, e.deprel
            applied += 1
            touched.add(sid)
    session.flush()
    for sid in touched:
        _update_metrics(session, sid)
    return applied


def _update_metrics(session: Session, sentence_id: int):
    from app.services.matching import load_sentences

    sent = load_sentences(session, [sentence_id]).get(sentence_id)
    if sent is None:
        return None
    an = SentenceAnalysis(sent)
    m = an.metrics()
    session.execute(update(SentenceRow).where(SentenceRow.id == sentence_id).values(
        tree_depth=m["tree_depth"], n_clauses=m["n_clauses"], sent_type=m["sent_type"], has_homogeneous=m["has_homogeneous"],
        has_participial=m["has_participial"], has_gerund=m["has_gerund"]))
    return sent, an


def refresh_sentence(session: Session, sentence_id: int) -> int:
    """Метрики предложения и вхождения всех словарей в нём заново (как при сопоставлении). Возвращает число вхождений."""
    from app.services.dictionaries import scope_text_ids
    from app.services.matching import entry_token_filter, flag_name_hits, relation_row, REL_COLS
    from app.services.phrases import contains_head
    from app.services.unlink import is_unlinked, unlinked_pairs

    res = _update_metrics(session, sentence_id)
    row = session.get(SentenceRow, sentence_id)
    if res is None or row is None or row.parse_level != PARSE_PRECISE:  # вхождения ищутся только в точном разборе
        session.commit()
        return 0
    sent, an = res
    lemmas = {norm(t.lemma) for t in sent.tokens}
    forms = {norm(t.form) for t in sent.tokens}
    pairs = unlinked_pairs(session, [sentence_id]).get(sentence_id)
    total, dict_ids = 0, []
    for d in session.scalars(select(Dictionary)):
        scope = scope_text_ids(session, d)
        if scope is not None and row.text_id not in scope:
            continue
        done = (d.settings or {}).get("matched_texts")
        if done is not None and row.text_id not in done:  # словарь с этим текстом ещё не сопоставлялся
            continue
        marks = {(e, t): v for e, t, v in session.execute(select(Hit.entry_id, Hit.token_id, Hit.is_valid).where(
            Hit.dictionary_id == d.id, Hit.sentence_id == sentence_id, Hit.is_valid.is_not(None)))}
        session.execute(delete(Hit).where(Hit.dictionary_id == d.id, Hit.sentence_id == sentence_id))
        settings = dict(d.settings or {})
        for e in d.entries:
            if e.match_mode == "forms":
                if not (word_forms(e.lemma, e.upos) & forms):
                    continue
            elif e.lemma_norm not in lemmas:
                continue
            rows = session.execute(select(TokenRow.id, TokenRow.idx, TokenRow.form).where(
                entry_token_filter(e, settings), TokenRow.sentence_id == sentence_id)).all()
            for r in rows:
                if e.phrase and not contains_head(sent.tokens, e.phrase["slots"], e.phrase["head"], r.idx):
                    continue
                role = an.role(r.idx)
                h = Hit(entry_id=e.id, dictionary_id=d.id, token_id=r.id, sentence_id=sentence_id, text_id=row.text_id,
                        token_idx=r.idx, form=r.form, role=role.role, role_subtype=role.subtype[:64] or None,
                        role_detail=role.detail[:64] or None, in_coord_group=an.group_of(r.idx) is not None,
                        syntax_version=SYNTAX_VERSION, is_valid=marks.get((e.id, r.id)))
                session.add(h)
                session.flush()
                for rel in an.relations(r.idx):
                    if not is_unlinked(pairs, r.idx, rel.related_idx):
                        session.add(TokenRelation(**dict(zip(REL_COLS, relation_row(h.id, d.id, e.id, row.text_id, rel)))))
                total += 1
        dict_ids.append(d.id)
    session.flush()
    if dict_ids:
        flag_name_hits(session, dict_ids, [row.text_id])
    session.commit()
    return total
