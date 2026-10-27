"""вхождения внутри любых имён собственных (dictionary_hits.in_propn): персоны, места, организации, а также
прилагательное с заглавной не в начале предложения («Чёрное море», «Белый дом») — не считаются при «Имена
собственные: не учитывать»

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-27
"""
import sqlalchemy as sa
from alembic import op

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('dictionary_hits', sa.Column('in_propn', sa.Boolean, nullable=False, server_default=sa.false()))
    # то же условие, что matching.PROPN_SQL (скопировано: миграция не зависит от кода приложения)
    op.execute("""UPDATE dictionary_hits h SET in_propn = (
        NOT EXISTS (SELECT 1 FROM dictionary_entries de WHERE de.id = h.entry_id
                    AND (de.upos = 'PROPN' OR substr(de.surface, 1, 1) <> lower(substr(de.surface, 1, 1)))) AND (
          EXISTS (SELECT 1 FROM entities e JOIN tokens t ON t.id = h.token_id
                  LEFT JOIN entity_overrides o ON o.norm = e.norm
                  WHERE e.sentence_id = h.sentence_id AND t.start_char >= e.start_char AND t.end_char <= e.end_char
                    AND coalesce(o.hidden, false) = false)
          OR EXISTS (SELECT 1 FROM tokens t WHERE t.id = h.token_id AND t.upos = 'ADJ'
                     AND substr(t.form, 1, 1) <> lower(substr(t.form, 1, 1))
                     AND EXISTS (SELECT 1 FROM tokens p WHERE p.sentence_id = t.sentence_id AND p.idx = t.idx - 1
                                 AND p.upos <> 'PUNCT'))))""")


def downgrade() -> None:
    op.drop_column('dictionary_hits', 'in_propn')
