#!/bin/bash
# Установка обновления на сервере БЕЗ интернета — запускать из папки пакета (с флешки):
#   ./offline-install.sh [ПАПКА_ПРОЕКТА] [--https]
# ПАПКА_ПРОЕКТА — где стоит Quotverba (по умолчанию /opt/quotverba); --https — с контейнером Caddy (профиль prod).
# Что делает: проверяет файлы → резервная копия базы → загружает образы → обновляет код (.env, копии, сертификаты,
# данные не трогает) → модели Stanza (если есть в пакете) → перезапуск без сборки и скачивания.
# Миграции базы применяются сами при запуске веб-контейнера.
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
APP="/opt/quotverba"; PROFILE=""
for a in "$@"; do
  case "$a" in
    --https) PROFILE="--profile prod" ;;
    -*) echo "Неизвестный параметр: $a"; exit 1 ;;
    *) APP="$a" ;;
  esac
done
[ -d "$APP" ] || { echo "Нет папки проекта $APP (укажите её первым параметром)"; exit 1; }
cat "$PKG/ВЕРСИЯ.txt" 2>/dev/null || true

echo "== 1/6 проверка файлов пакета"
(cd "$PKG" && sha256sum -c SHA256SUMS)

cd "$APP"
DC="docker compose -f docker-compose.yml -f deploy/compose.offline.yml $PROFILE"
echo "== 2/6 резервная копия базы перед обновлением"
if [ -f .env ] && docker compose ps --status running db 2>/dev/null | grep -q db; then
  ./deploy/backup.sh || { echo "Резервная копия не сделана — обновление остановлено"; exit 1; }
else
  echo "база не запущена — копия пропущена (первая установка?)"
fi

echo "== 3/6 загрузка образов (несколько минут)"
if [ -f "$PKG/images.tar.gz" ]; then docker load -i "$PKG/images.tar.gz"; else docker load -i "$PKG/images.tar"; fi  # .tar — пакет из Windows

echo "== 4/6 обновление кода в $APP"
tar xzf "$PKG/code.tar.gz" -C "$APP"
chmod +x deploy/*.sh
[ -f .env ] || { cp .env.example .env; echo "Создан .env из образца — заполните пароли (DB_PASSWORD, ADMIN_PASSWORD, SECRET_KEY) и запустите скрипт снова"; exit 1; }

if [ -f "$PKG/stanza_models.tar.gz" ]; then
  echo "== 5/6 модели Stanza"
  $DC run --rm --no-deps -v "$PKG":/pkg:ro --entrypoint tar worker xzf /pkg/stanza_models.tar.gz -C /models/stanza
else
  echo "== 5/6 моделей в пакете нет — остаются те, что уже на сервере"
fi

echo "== 6/6 перезапуск"
$DC up -d --no-build --remove-orphans
sleep 5
$DC ps
echo "Готово: версия — внизу бокового меню; журнал — docker compose logs -f web worker"
