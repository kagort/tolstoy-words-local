"""словосочетания в словаре (dictionary_entries.phrase) и координаты мест (places)

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-23
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # {"slots": [...], "head": k} — запись словаря из нескольких слов; вхождение привязано к главному слову
    op.add_column('dictionary_entries', sa.Column('phrase', JSONB, nullable=True))
    # координаты мест для географической карты: найдены в справочнике (auto), поставлены вручную (manual)
    # или помечены «не на карте» (none — выдуманное место, ошибка распознавания)
    op.create_table(
        'places',
        sa.Column('norm', sa.Text, primary_key=True),
        sa.Column('lat', sa.Float, nullable=True),
        sa.Column('lon', sa.Float, nullable=True),
        sa.Column('source', sa.String(16), nullable=False, server_default='auto'),
        sa.Column('label', sa.Text, nullable=True),
        sa.Column('country', sa.String(8), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('places')
    op.drop_column('dictionary_entries', 'phrase')
