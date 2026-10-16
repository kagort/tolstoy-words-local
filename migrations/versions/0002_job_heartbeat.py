"""jobs: heartbeat, attempts, cancel_requested

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('jobs', sa.Column('attempts', sa.Integer(), server_default='0', nullable=False))
    op.add_column('jobs', sa.Column('cancel_requested', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    op.drop_column('jobs', 'cancel_requested')
    op.drop_column('jobs', 'attempts')
    op.drop_column('jobs', 'heartbeat_at')
