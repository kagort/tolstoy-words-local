"""entities, entity_overrides (имена собственные)

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-18
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'entities',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('text_id', sa.Integer(), nullable=False),
        sa.Column('sentence_id', sa.BigInteger(), nullable=False),
        sa.Column('etype', sa.String(length=8), nullable=False),
        sa.Column('start_char', sa.Integer(), nullable=False),
        sa.Column('end_char', sa.Integer(), nullable=False),
        sa.Column('surface', sa.Text(), nullable=False),
        sa.Column('norm', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['text_id'], ['texts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['sentence_id'], ['sentences.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_entities_text_type', 'entities', ['text_id', 'etype'])
    op.create_index('ix_entities_norm', 'entities', ['norm'])
    op.create_index('ix_entities_sentence_id', 'entities', ['sentence_id'])
    op.create_table(
        'entity_overrides',
        sa.Column('norm', sa.Text(), nullable=False),
        sa.Column('etype', sa.String(length=8), nullable=True),
        sa.Column('canonical', sa.Text(), nullable=True),
        sa.Column('hidden', sa.Boolean(), server_default='false', nullable=False),
        sa.PrimaryKeyConstraint('norm'),
    )


def downgrade() -> None:
    op.drop_table('entity_overrides')
    op.drop_index('ix_entities_sentence_id', table_name='entities')
    op.drop_index('ix_entities_norm', table_name='entities')
    op.drop_index('ix_entities_text_type', table_name='entities')
    op.drop_table('entities')
