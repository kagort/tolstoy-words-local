"""вхождения внутри имён (dictionary_hits.in_name), псевдонимы персонажей (entity_aliases), вид места (places.kind),
скорость задачи для оценки оставшегося времени (jobs.eta)

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-24
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # слово словаря внутри имени собственного («Любовь Андреевна» для слова «любовь») — по умолчанию не считается
    op.add_column('dictionary_hits', sa.Column('in_name', sa.Boolean, nullable=False, server_default=sa.false()))
    # уже найденные вхождения: помечаем стоящие внутри извлечённых имён (как matching.flag_name_hits)
    op.execute("""
        UPDATE dictionary_hits h SET in_name = true
        WHERE EXISTS (
            SELECT 1 FROM entities e
            JOIN tokens t ON t.id = h.token_id
            LEFT JOIN entity_overrides o ON o.norm = e.norm
            WHERE e.sentence_id = h.sentence_id AND t.start_char >= e.start_char AND t.end_char <= e.end_char
              AND coalesce(o.hidden, false) = false)""")
    # варианты имени одного персонажа; text_id — слияние только в этом тексте (одноимённые герои разных романов)
    op.create_table(
        'entity_aliases',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('norm', sa.Text, nullable=False),
        sa.Column('canonical', sa.Text, nullable=False),
        sa.Column('text_id', sa.Integer, sa.ForeignKey('texts.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_entity_aliases_norm', 'entity_aliases', ['norm'])
    op.create_index('ux_entity_aliases_norm_text', 'entity_aliases', ['norm', sa.text('coalesce(text_id, 0)')], unique=True)
    # вид места, заданный пользователем: city | country | region | river | mountain | sea | fictional
    op.add_column('places', sa.Column('kind', sa.String(16), nullable=True))
    # {"t": время, "p": прогресс, "rate": сглаженная скорость в долях в секунду} — см. jobs.set_progress
    op.add_column('jobs', sa.Column('eta', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'eta')
    op.drop_column('places', 'kind')
    op.drop_index('ux_entity_aliases_norm_text', table_name='entity_aliases')
    op.drop_index('ix_entity_aliases_norm', table_name='entity_aliases')
    op.drop_table('entity_aliases')
    op.drop_column('dictionary_hits', 'in_name')
