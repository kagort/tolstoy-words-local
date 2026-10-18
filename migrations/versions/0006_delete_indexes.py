"""индексы для быстрого удаления текстов (dictionary_hits.token_id, dictionary_hits.text_id)

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-18
"""
from alembic import op

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # без индекса по token_id удаление каждого токена проверяет всю таблицу вхождений (внешний ключ ON DELETE CASCADE)
    op.create_index('ix_hits_token_id', 'dictionary_hits', ['token_id'])
    op.create_index('ix_hits_text_id', 'dictionary_hits', ['text_id'])
    op.create_index('ix_rel_text_id', 'token_relations', ['text_id'])


def downgrade() -> None:
    op.drop_index('ix_rel_text_id', table_name='token_relations')
    op.drop_index('ix_hits_text_id', table_name='dictionary_hits')
    op.drop_index('ix_hits_token_id', table_name='dictionary_hits')
