#!/bin/sh
# Точка входа контейнера: web | worker | любая другая команда
set -e
case "$1" in
  web)
    alembic upgrade head
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers "${WEB_WORKERS:-2}" --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    exec python -m app.worker
    ;;
  *)
    exec "$@"
    ;;
esac
