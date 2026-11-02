"""отзывы тестировщиков (feedback) и «кто загрузил» у текстов, корпусов и словарей (created_by)

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-02
"""
import sqlalchemy as sa
from alembic import op

revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ('texts', 'corpora', 'dictionaries'):
        op.add_column(table, sa.Column('created_by', sa.String(100), nullable=True))
    op.create_table(
        'feedback',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('author', sa.String(100)),
        sa.Column('page', sa.String(1000)),
        sa.Column('kind', sa.String(16), nullable=False, server_default='other'),
        sa.Column('text', sa.Text, nullable=False),
        sa.Column('status', sa.String(16), nullable=False, server_default='new'),
        sa.Column('reply', sa.Text),
    )
    op.create_index('ix_feedback_created_at', 'feedback', ['created_at'])


def downgrade() -> None:
    op.drop_table('feedback')
    for table in ('texts', 'corpora', 'dictionaries'):
        op.drop_column(table, 'created_by')
