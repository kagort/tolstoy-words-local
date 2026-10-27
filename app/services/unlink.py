"""Связи, убранные вручную (модель RelationOverride): «пчёлы» не подлежащее при «взрывают», если однородный ряд
сказуемых на самом деле относится к другому, опущенному подлежащему.

Отметка снимает все связи между двумя словами предложения в обе стороны: на странице предложения, в сохранённых связях
вхождений (профиль слова, коллокации, отбор по связи) и при следующих пересчётах (matching._match_entries).
"""

from __future__ import annotations

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.orm import Session

from app.models import Hit, RelationOverride, TokenRelation


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def unlinked_pairs(session: Session, sentence_ids) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    ids = list(sentence_ids)
    if not ids:
        return out
    for sid, a, b in session.execute(select(RelationOverride.sentence_id, RelationOverride.a_idx, RelationOverride.b_idx)
                                     .where(RelationOverride.sentence_id.in_(ids))):
        out.setdefault(sid, set()).add((a, b))
    return out


def is_unlinked(pairs: set[tuple[int, int]] | None, i: int, j: int) -> bool:
    return bool(pairs) and _pair(i, j) in pairs


def unlink(session: Session, sentence_id: int, i: int, j: int) -> int:
    """Убрать связь слов i и j; возвращает, сколько сохранённых связей вхождений удалено."""
    a, b = _pair(i, j)
    if not session.scalar(select(RelationOverride.id).where(RelationOverride.sentence_id == sentence_id,
                                                            RelationOverride.a_idx == a, RelationOverride.b_idx == b)):
        session.add(RelationOverride(sentence_id=sentence_id, a_idx=a, b_idx=b))
    hits = select(Hit.id).where(Hit.sentence_id == sentence_id, Hit.token_idx.in_([a, b]))
    res = session.execute(delete(TokenRelation).where(
        TokenRelation.hit_id.in_(hits),
        or_(and_(TokenRelation.related_idx == b, TokenRelation.hit_id.in_(hits.where(Hit.token_idx == a))),
            and_(TokenRelation.related_idx == a, TokenRelation.hit_id.in_(hits.where(Hit.token_idx == b))))))
    session.commit()
    return res.rowcount or 0


def relink(session: Session, sentence_id: int, i: int, j: int) -> int:
    """Вернуть связь: снять отметку и заново записать связи вхождений между i и j (как при пересчёте)."""
    from app.nlp.syntax import SentenceAnalysis
    from app.services.matching import REL_COLS, load_sentences, relation_row

    a, b = _pair(i, j)
    session.execute(delete(RelationOverride).where(RelationOverride.sentence_id == sentence_id,
                                                   RelationOverride.a_idx == a, RelationOverride.b_idx == b))
    hits = session.execute(select(Hit).where(Hit.sentence_id == sentence_id, Hit.token_idx.in_([a, b]))).scalars().all()
    added = 0
    if hits:
        sent = load_sentences(session, [sentence_id])[sentence_id]
        an = SentenceAnalysis(sent)
        for h in hits:
            other = b if h.token_idx == a else a
            for rel in an.relations(h.token_idx):
                if rel.related_idx == other:
                    session.add(TokenRelation(**dict(zip(REL_COLS, relation_row(h.id, h.dictionary_id, h.entry_id, h.text_id, rel)))))
                    added += 1
    session.commit()
    return added
