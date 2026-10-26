"""связи, убранные вручную на странице предложения (relation_overrides)

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-26
"""
import sqlalchemy as sa
from alembic import op

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'relation_overrides',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('sentence_id', sa.BigInteger, sa.ForeignKey('sentences.id', ondelete='CASCADE'), nullable=False),
        sa.Column('a_idx', sa.SmallInteger, nullable=False),
        sa.Column('b_idx', sa.SmallInteger, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('sentence_id', 'a_idx', 'b_idx'),
    )
    op.create_index('ix_relation_overrides_sentence_id', 'relation_overrides', ['sentence_id'])


def downgrade() -> None:
    op.drop_table('relation_overrides')
