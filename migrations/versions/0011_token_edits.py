"""ручные правки разбора на странице предложения (token_edits)

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-29
"""
import sqlalchemy as sa
from alembic import op

revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'token_edits',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('text_id', sa.Integer, sa.ForeignKey('texts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sent_idx', sa.Integer, nullable=False),
        sa.Column('idx', sa.SmallInteger, nullable=False),
        sa.Column('form', sa.Text, nullable=False),
        sa.Column('upos', sa.String(8), nullable=False),
        sa.Column('head', sa.SmallInteger, nullable=False),
        sa.Column('deprel', sa.String(32), nullable=False),
        sa.Column('orig_upos', sa.String(8), nullable=False),
        sa.Column('orig_head', sa.SmallInteger, nullable=False),
        sa.Column('orig_deprel', sa.String(32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('text_id', 'sent_idx', 'idx'),
    )
    op.create_index('ix_token_edits_text_id', 'token_edits', ['text_id'])


def downgrade() -> None:
    op.drop_table('token_edits')
