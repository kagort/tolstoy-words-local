#!/bin/sh
# Восстановление из копии: ./deploy/restore.sh backups/tolstoy-2026-10-13_0315.dump
# ВНИМАНИЕ: текущее содержимое базы будет заменено.
set -e
cd "$(dirname "$0")/.."
. ./.env 2>/dev/null || true
[ -f "$1" ] || { echo "Укажите файл копии"; exit 1; }
docker compose stop web worker
docker compose exec -T db pg_restore -U "${DB_USER:-postgres}" -d "${DB_NAME:-tolstoy}" --clean --if-exists < "$1"
docker compose start web worker
echo "Восстановлено из $1"
