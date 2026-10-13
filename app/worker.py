"""Фоновый обработчик задач: python -m app.worker"""

import logging
import time

from app.config import settings
from app.services.jobs import worker_loop


def wait_for_db(attempts: int = 60) -> None:
    """Ждём, пока БД поднимется и веб-контейнер применит миграции."""
    from sqlalchemy import text

    from app.db import engine

    for i in range(attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM jobs LIMIT 1"))
            return
        except Exception:  # noqa: BLE001
            logging.info("Жду базу данных (%s/%s)…", i + 1, attempts)
            time.sleep(3)
    raise SystemExit("База данных недоступна")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    wait_for_db()
    # модель загружаем заранее, чтобы первая задача не ждала
    from app.nlp.parsers import get_parser

    get_parser()
    worker_loop(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
