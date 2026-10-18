"""sentences.parse_level (двухступенчатая обработка)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-18
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # уже разобранные предложения были разобраны основным парсером — уровень 2 (точный)
    op.add_column('sentences', sa.Column('parse_level', sa.SmallInteger(), server_default='2', nullable=False))
    op.create_index('ix_sentences_text_level', 'sentences', ['text_id', 'parse_level'])


def downgrade() -> None:
    op.drop_index('ix_sentences_text_level', table_name='sentences')
    op.drop_column('sentences', 'parse_level')
