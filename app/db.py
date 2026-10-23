"""Подключение к PostgreSQL (SQLAlchemy 2)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _connect_args() -> dict:
    """Даты из базы приходят в часовом поясе TIMEZONE (по умолчанию в контейнере PostgreSQL — UTC)."""
    tz = (settings.timezone or "").strip()
    if not tz or not all(c.isalnum() or c in "/_+-" for c in tz):
        return {}
    return {"options": f"-c timezone={tz}"}


engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True, future=True, connect_args=_connect_args())
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def get_session() -> Iterator[Session]:
    """Зависимость FastAPI: одна сессия на запрос."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
