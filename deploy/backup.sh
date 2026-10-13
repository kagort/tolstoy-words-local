#!/bin/sh
# Резервная копия базы: ./deploy/backup.sh  (запускать из папки проекта)
# Для ежедневного запуска на VPS добавьте в crontab (crontab -e):
#   15 3 * * * cd /opt/tolstoy-words && ./deploy/backup.sh >> backups/backup.log 2>&1
set -e
cd "$(dirname "$0")/.."
. ./.env 2>/dev/null || true
mkdir -p backups
FILE="backups/tolstoy-$(date +%Y-%m-%d_%H%M).dump"
docker compose exec -T db pg_dump -U "${DB_USER:-postgres}" -d "${DB_NAME:-tolstoy}" -Fc > "$FILE"
echo "$(date '+%F %T') копия сохранена: $FILE ($(du -h "$FILE" | cut -f1))"
# храним копии 14 дней
find backups -name 'tolstoy-*.dump' -mtime +14 -delete
