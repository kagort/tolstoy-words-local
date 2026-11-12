#!/bin/bash
# Пакет обновления для сервера без интернета — запускать на компьютере С интернетом и Docker, в папке проекта:
#   ./deploy/offline-pack.sh [ПАПКА] [--with-models] [--app-only]
# ПАПКА (по умолчанию ./offline-update) потом целиком копируется на флешку.
#   --with-models  добавить модели Stanza (~600 МБ; нужны при первой установке или если на сервере их ещё нет)
#   --app-only     только образ приложения, без postgres и caddy (если они на сервере уже есть)
# Установка на сервере: deploy/offline-install.sh (лежит в пакете) — см. docs/DEPLOY.md §2.8.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="offline-update"; MODELS=0; APP_ONLY=0
for a in "$@"; do
  case "$a" in
    --with-models) MODELS=1 ;;
    --app-only) APP_ONLY=1 ;;
    -*) echo "Неизвестный параметр: $a"; exit 1 ;;
    *) OUT="$a" ;;
  esac
done
mkdir -p "$OUT"
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' app/__init__.py)
echo "== Quotverba $VERSION: пакет обновления в $OUT"

echo "== 1/4 сборка образа приложения (нужен интернет: пакеты Python, PyTorch)"
docker compose build web
IMAGES="quotverba:latest"
if [ "$APP_ONLY" = 0 ]; then
  echo "== 2/4 образы базы данных и HTTPS-прокси"
  docker pull postgres:16
  docker pull caddy:2
  IMAGES="$IMAGES postgres:16 caddy:2"
else
  echo "== 2/4 пропущено (--app-only)"
fi

echo "== 3/4 сохранение образов ($IMAGES) — несколько минут"
docker save $IMAGES | gzip > "$OUT/images.tar.gz"
git archive --format=tar.gz -o "$OUT/code.tar.gz" HEAD
cp deploy/offline-install.sh "$OUT/"

if [ "$MODELS" = 1 ]; then
  echo "== 4/4 модели Stanza из тома Docker"
  VOL=$(docker volume ls -q --filter label=com.docker.compose.volume=stanza_models | head -1)
  if [ -z "$VOL" ]; then
    echo "Том с моделями не найден: сначала запустите приложение здесь (docker compose up -d) и дождитесь задачи"
    echo "«Загрузка моделей Stanza» на странице «Задачи», затем повторите с --with-models."; exit 1
  fi
  docker run --rm -v "$VOL":/m:ro -v "$PWD/$OUT":/out --entrypoint tar quotverba:latest czf /out/stanza_models.tar.gz -C /m .
else
  echo "== 4/4 модели Stanza не добавлены (на сервере уже есть — или добавьте --with-models)"
fi

{
  echo "Quotverba $VERSION, пакет от $(date '+%Y-%m-%d %H:%M'), коммит $(git rev-parse --short HEAD)"
  echo "Установка на сервере: sudo ./offline-install.sh /путь/к/проекту   (см. docs/DEPLOY.md §2.8)"
} > "$OUT/ВЕРСИЯ.txt"
(cd "$OUT" && sha256sum images.tar.gz code.tar.gz offline-install.sh $( [ -f stanza_models.tar.gz ] && echo stanza_models.tar.gz ) > SHA256SUMS)
du -sh "$OUT"
echo "Готово. Скопируйте папку $OUT на флешку целиком."
