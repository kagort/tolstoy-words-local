#!/bin/sh
# Автоматические резервные копии базы (контейнер backup в docker-compose.yml).
# Раз в сутки в BACKUP_HOUR (по TZ) — pg_dump в /backups (папка ./backups проекта); хранятся BACKUP_KEEP последних.
# Сразу после запуска, если копий ещё нет, — первая копия. Восстановление: ./deploy/restore.sh backups/ИМЯ.dump
HOUR="${BACKUP_HOUR:-3}"
KEEP="${BACKUP_KEEP:-7}"
mkdir -p /backups

dump() {
  f="/backups/tolstoy-$(date +%Y-%m-%d_%H%M).dump"
  if pg_dump -Fc -f "$f.part"; then
    mv "$f.part" "$f"
    echo "$(date '+%F %T') копия сохранена: $f ($(du -h "$f" | cut -f1))"
  else
    rm -f "$f.part"
    echo "$(date '+%F %T') ОШИБКА: копия не сохранена" >&2
  fi
  # оставить KEEP последних
  ls -1t /backups/tolstoy-*.dump 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do rm -f "$old"; done
}

until pg_isready -q; do sleep 5; done
ls /backups/tolstoy-*.dump >/dev/null 2>&1 || dump
while true; do
  today="$(date +%Y-%m-%d)"
  if [ "$(date +%H | sed 's/^0//')" -ge "$HOUR" ] && ! ls /backups/tolstoy-"$today"_*.dump >/dev/null 2>&1; then
    dump
  fi
  sleep 600
done
