"""Интеграционные тесты требуют PostgreSQL: задайте TEST_DATABASE_URL.

Пример: TEST_DATABASE_URL=postgresql://postgres:пароль@localhost:5432/tolstoy_test pytest
Без этой переменной интеграционные тесты пропускаются, модульные выполняются всегда.
"""

import os

TEST_DB = os.environ.get("TEST_DATABASE_URL")
if TEST_DB:
    os.environ["DATABASE_URL"] = TEST_DB
    os.environ.setdefault("NLP_BACKEND", "natasha")
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "test-password"
    os.environ["SECRET_KEY"] = "test-secret"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_noprop():
    """Режим «Имена собственные: не учитывать» задаётся на запрос (selection.from_params); тесты, вызывающие from_params
    напрямую в процессе тестов, не должны переносить его в следующие тесты."""
    yield
    from app.services import analytics

    analytics.set_noprop(False)
