# Quotverba

Веб-приложение для корпусного анализа лексики русских текстов: загрузка произвольных текстов и словарей,
морфологический и синтаксический разбор (Universal Dependencies + слой членов предложения и однородных рядов),
частоты, конкорданс, синтаксический профиль слова, сравнение корпусов (литература vs тексты нейросетей),
экспорт в XLSX / CSV / JSON / CoNLL-U.

## Быстрый старт (Windows + Docker Desktop)

```powershell
git clone https://github.com/kagort/tolstoy-words-local.git
cd tolstoy-words-local
Copy-Item .env.example .env      # впишите DB_PASSWORD, ADMIN_PASSWORD, SECRET_KEY
docker compose up -d --build
```
Откройте http://localhost:8000 и войдите (`admin` + `ADMIN_PASSWORD`).
Демо-данные: `docker compose run --rm worker python -m app.cli seed-demo`.

## Документация

| Документ | Для чего |
|---|---|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Руководство пользователя: как загружать тексты и словари, читать результаты, экспортировать |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Установка локально и на VPS (HTTPS, бэкапы, перенос базы) |
| [docs/TZ_MVP.md](docs/TZ_MVP.md) | Техническое задание |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Журнал разработки: что сделано, упрощения, бэкалог |

## Структура

```
app/            приложение
  nlp/          нормализация, сегментация, парсеры (Stanza / Natasha), синтаксический слой (syntax.py)
  services/     загрузка текстов, словари, сопоставление, аналитика, статистика, очередь задач
  web/          страницы, экспорт
  templates/    HTML-шаблоны
migrations/     миграции схемы БД (Alembic)
tests/          автотесты
deploy/         Caddy (HTTPS), entrypoint, бэкап/восстановление
data/raw/       корпус художественных текстов (метаданные — data/processed/dictexts.csv)
data/generated/ тексты, сгенерированные нейросетями
legacy/         прежняя версия проекта (скрипты, Flask-приложение) — для справки
```

## Разработка

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements-dev.txt
docker compose up -d db          # только база
alembic upgrade head
uvicorn app.main:app --reload    # и в другом окне: python -m app.worker
pytest
```
