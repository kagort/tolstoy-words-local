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
import shutil
import threading
import time

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from app.services.jobs import enqueue, set_progress

log = logging.getLogger(__name__)

PROCESSORS = "tokenize,pos,lemma,depparse"
ATTEMPTS = 5
EXPECTED_MB = 600  # примерный объём русских моделей — только для подписи и полосы прогресса
STALL_SECONDS = 180  # столько секунд без новых байтов — предупреждаем, что загрузка стоит


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


def hf_cache_dir() -> str:
    """Кэш huggingface_hub: туда Stanza сначала качает файл (*.incomplete), потом копирует в model_dir."""
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def downloaded_bytes() -> int:
    """Сколько уже скачано: готовые файлы моделей + недокачанные файлы в кэше huggingface_hub."""
    total = 0
    cache = os.path.abspath(hf_cache_dir())
    for root, dirs, files in os.walk(model_dir()):
        if os.path.abspath(root) == cache:  # кэш внутри папки моделей считаем отдельно, без двойного счёта
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != cache]
        total += sum(_size(os.path.join(root, f)) for f in files)
    for root, _dirs, files in os.walk(cache):
        total += sum(_size(os.path.join(root, f)) for f in files if f.endswith(".incomplete"))
    return total


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:  # файл переименовали или удалили между listdir и stat
        return 0


class _Monitor(threading.Thread):
    """Раз в несколько секунд пишет в задачу, сколько скачано и с какой скоростью (и тем самым подаёт «пульс»)."""

    def __init__(self, job_id: int, attempt: int, every: float = 5.0):
        super().__init__(daemon=True)
        self.job_id, self.attempt, self.every = job_id, attempt, every
        self.stop = threading.Event()

    def run(self) -> None:
        prev, prev_t = downloaded_bytes(), time.monotonic()
        last_growth = prev_t
        while not self.stop.wait(self.every):
            cur, t = downloaded_bytes(), time.monotonic()
            speed = max(cur - prev, 0) / max(t - prev_t, 1e-6)
            if cur > prev:
                last_growth = t
            prev, prev_t = cur, t
            mb = cur / 2**20
            msg = (f"Попытка {self.attempt} из {ATTEMPTS}: скачано {mb:.0f} МБ (всего около {EXPECTED_MB} МБ), "
                   f"скорость {speed / 2**20:.2f} МБ/с")
            idle = t - last_growth
            if idle >= STALL_SECONDS:
                msg += (f". Уже {idle / 60:.0f} мин нет новых данных — похоже, соединение с huggingface.co не идёт "
                        "(см. «Если что-то не работает» в docs/DEPLOY.md)")
            try:
                set_progress(self.job_id, min(mb / EXPECTED_MB, 0.99), msg)
            except Exception:  # noqa: BLE001 — в том числе запрос отмены: его обработает основной поток
                return


def run_download(job_id: int | None) -> None:
    """Задача «models»: скачивание с повторами (10 с, 30 с, 1 мин, 2 мин между попытками)."""
    import stanza

    last = None
    for attempt in range(1, ATTEMPTS + 1):
        if job_id:
            set_progress(job_id, (attempt - 1) / ATTEMPTS,
                         f"Скачивание моделей Stanza, попытка {attempt} из {ATTEMPTS} (уже скачанные файлы пропускаются)")
        monitor = _Monitor(job_id, attempt) if job_id else None
        if monitor:
            monitor.start()
        try:
            stanza.download("ru", model_dir=model_dir(), processors=PROCESSORS, logging_level="WARN")
            if stanza_ready():
                from app.nlp import parsers

                parsers._PARSERS.pop("stanza", None)
                log.info("Модели Stanza скачаны в %s", model_dir())
                if os.path.abspath(hf_cache_dir()).startswith(os.path.abspath(model_dir()) + os.sep):
                    shutil.rmtree(hf_cache_dir(), ignore_errors=True)  # копии файлов уже в model_dir
                requeue_waiting()
                return
            last = RuntimeError("скачивание завершилось, но файлов модели нет")
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("Попытка %s скачать модели Stanza не удалась: %s", attempt, exc)
        finally:
            if monitor:
                monitor.stop.set()
                monitor.join(timeout=10)
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
