"""Настройки приложения: читаются из переменных окружения и файла .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # База данных: либо готовый DATABASE_URL, либо части по отдельности
    database_url: str = ""
    db_user: str = "postgres"
    db_password: str = ""
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "tolstoy"

    # Веб
    secret_key: str = "change-me"
    admin_username: str = "admin"
    admin_password: str = ""
    # общий пароль тестировщиков (пусто — вход только администратора): с ним входят под своим именем (любое имя),
    # имя сохраняется у загруженных текстов, корпусов, словарей и у отзывов
    tester_password: str = ""
    upload_max_mb: int = 50
    site_title: str = "Quotverba"
    # интерфейс по умолчанию: v2 — новый (боковое меню, токены DESIGN_TOKENS.md), classic — прежний (откат: UI_DEFAULT=classic);
    # каждый пользователь может переключить у себя (ссылка внизу страницы, cookie qv_ui)
    ui_default: str = "v2"
    # часовой пояс для дат на страницах и в выгрузках (база хранит время с поясом, показывается в этом)
    timezone: str = "Europe/Moscow"

    # NLP
    nlp_backend: str = "stanza"  # stanza | natasha
    stanza_dir: str = ""
    parse_batch_size: int = 64
    # точный разбор (Stanza): предложений за один вызов — крупнее пачка, меньше накладных расходов; разбор тот же
    precise_batch_size: int = 256
    # потоков PyTorch на один обработчик: 0 — по числу ядер, доступных контейнеру (учитывает ограничение Docker cpus);
    # при нескольких обработчиках (docker compose up --scale worker=N) — ядра / N
    torch_threads: int = 0
    # папка с резервными копиями базы (контейнер backup, docker-compose.yml) — для строки «последняя копия» на «Задачах»
    backup_dir: str = "/app/backups"
    # two_stage — весь текст быстро разбирается FAST_PARSER (Natasha), а основным парсером (NLP_BACKEND, Stanza)
    #             только предложения со словами словарей и по кнопке «Точный разбор»; full — всё основным парсером
    processing_mode: str = "two_stage"
    fast_parser: str = "natasha"
    # после разбора текста автоматически извлекать имена собственные (персоны, места, организации)
    ner_on_ingest: bool = True

    # Фоновый обработчик
    worker_poll_seconds: float = 2.0
    # задача без «пульса» дольше этого срока считается брошенной (обработчик упал) и возвращается в очередь
    job_stale_minutes: float = 5.0
    # после стольких падений обработчика на одной задаче она помечается ошибкой (защита от бесконечного цикла)
    job_max_attempts: int = 3

    @property
    def two_stage(self) -> bool:
        """Двухступенчатая обработка включена и имеет смысл (быстрый и основной парсеры различаются)."""
        return self.processing_mode.lower() == "two_stage" and self.fast_parser.lower() != self.nlp_backend.lower()

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            url = self.database_url
            if url.startswith("postgresql://"):
                url = "postgresql+psycopg://" + url[len("postgresql://") :]
            return url
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
