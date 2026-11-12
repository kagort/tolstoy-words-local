"""срочность отзыва (feedback.priority: urgent | important | normal)

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-12
"""
import sqlalchemy as sa
from alembic import op

revision = '0013'
down_revision = '0012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('feedback', sa.Column('priority', sa.String(16), nullable=False, server_default='normal'))


def downgrade() -> None:
    op.drop_column('feedback', 'priority')
