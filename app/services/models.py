"""Модели Stanza: скачиваются не при сборке образа, а фоновой задачей «models» в том Docker (/models/stanza).

Почему так: при сборке образа скачивание сотен мегабайт с HuggingFace на медленном или нестабильном
соединении (или без доступа к huggingface.co) роняло всю сборку. Теперь образ собирается без моделей,
а обработчик при первом запуске скачивает их сам — с повторами; уже скачанные файлы повторно не качаются
(Stanza сверяет контрольные суммы), поэтому каждая попытка продолжает с места обрыва. Ошибка видна на
странице «Задачи» с кнопкой «Повторить».

Обход блокировок и медленной сети (переменные в .env):
    HTTPS_PROXY=http://адрес:порт           — прокси для скачивания;
    HF_ENDPOINT=https://hf-mirror.com       — зеркало HuggingFace;
    STANZA_RESOURCES_URL=...                — другой адрес списка моделей (по умолчанию raw.githubusercontent.com).
"""

from __future__ import annotations

import logging
import os
import time

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from app.services.jobs import enqueue, set_progress

log = logging.getLogger(__name__)

PROCESSORS = "tokenize,pos,lemma,depparse"
ATTEMPTS = 5


def model_dir() -> str:
    return settings.stanza_dir or "/models/stanza"


def stanza_ready() -> bool:
    """Скачаны ли русские модели Stanza (список моделей и папка ru с разбором зависимостей)."""
    d = model_dir()
    return os.path.exists(os.path.join(d, "resources.json")) and os.path.isdir(os.path.join(d, "ru", "depparse"))


def needs_models() -> bool:
    return settings.nlp_backend.lower() == "stanza" and not stanza_ready()


def ensure_download_job() -> int | None:
    """Ставит задачу скачивания моделей, если они нужны и задачи ещё нет. Возвращает id задачи."""
    if not needs_models():
        return None
    with SessionLocal() as s:
        active = s.scalar(select(Job.id).where(Job.kind == "models", Job.status.in_(["queued", "running"])))
        if active:
            return active
        job = enqueue(s, "models", {}, title="Загрузка моделей Stanza (один раз, ~600 МБ)")
        s.commit()
        return job.id


def run_download(job_id: int | None) -> None:
    """Задача «models»: скачивание с повторами (10 с, 30 с, 1 мин, 2 мин между попытками)."""
    import stanza

    last = None
    for attempt in range(1, ATTEMPTS + 1):
        if job_id:
            set_progress(job_id, (attempt - 1) / ATTEMPTS,
                         f"Скачивание моделей Stanza, попытка {attempt} из {ATTEMPTS} (уже скачанные файлы пропускаются)")
        try:
            stanza.download("ru", model_dir=model_dir(), processors=PROCESSORS, logging_level="WARN")
            if stanza_ready():
                from app.nlp import parsers

                parsers._PARSERS.pop("stanza", None)
                log.info("Модели Stanza скачаны в %s", model_dir())
                requeue_waiting()
                return
            last = RuntimeError("скачивание завершилось, но файлов модели нет")
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("Попытка %s скачать модели Stanza не удалась: %s", attempt, exc)
        if attempt < ATTEMPTS:
            time.sleep((10, 30, 60, 120)[attempt - 1])
    raise RuntimeError(
        f"Не удалось скачать модели Stanza за {ATTEMPTS} попыток: {last}. Проверьте интернет; если huggingface.co недоступен, "
        "включите VPN или задайте в .env HF_ENDPOINT=https://hf-mirror.com (зеркало) или HTTPS_PROXY=… и перезапустите "
        "контейнеры (docker compose up -d), затем нажмите «Повторить». Пока моделей нет, тексты разбираются только Natasha."
    )


MISSING_MARK = "Модели Stanza ещё не скачаны"


def requeue_waiting() -> int:
    """Задачи, упавшие из-за отсутствия моделей, после их загрузки снова ставятся в очередь."""
    n = 0
    with SessionLocal() as s:
        for job in s.scalars(select(Job).where(Job.status == "error", Job.error.like(f"%{MISSING_MARK}%"))):
            enqueue(s, job.kind, job.params, title=job.title)
            s.delete(job)
            n += 1
        s.commit()
    if n:
        log.info("После загрузки моделей возвращено в очередь задач: %s", n)
    return n
