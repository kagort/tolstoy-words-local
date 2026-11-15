# Установка обновления на сервере с Windows БЕЗ интернета (Docker Desktop запущен). Из папки пакета:
#   powershell -ExecutionPolicy Bypass -File .\offline-install.ps1 -App "C:\quotverba" [-Https]
# -App — папка, где стоит Quotverba (в ней docker-compose.yml и .env); -Https — с контейнером Caddy (профиль prod).
# Шаги: проверка файлов -> резервная копия базы -> загрузка образов -> обновление кода (.env, backups, certs и данные
# не трогаются) -> модели Stanza (если есть в пакете) -> перезапуск без сборки и скачивания. Миграции базы применяются
# сами при запуске. Для Linux — offline-install.sh. См. docs/DEPLOY.md, раздел 2.8.
param([string]$App = "", [switch]$Https)
$ErrorActionPreference = "Stop"
$Pkg = $PSScriptRoot
function Run-Docker { & docker.exe @args; if ($LASTEXITCODE -ne 0) { throw "Ошибка команды: docker $($args -join ' ')" } }

if (-not $App) { throw 'Укажите папку проекта: -App "C:\путь\к\quotverba"' }
if (-not (Test-Path (Join-Path $App "docker-compose.yml"))) { throw "В папке $App нет docker-compose.yml - это не папка Quotverba" }
$App = (Resolve-Path $App).Path
if (Test-Path (Join-Path $Pkg "ВЕРСИЯ.txt")) { Get-Content (Join-Path $Pkg "ВЕРСИЯ.txt") -Encoding UTF8 | Select-Object -First 1 }

Write-Host "== 1/6 проверка файлов пакета"
foreach ($line in Get-Content (Join-Path $Pkg "SHA256SUMS") -Encoding UTF8) {
  if (-not $line.Trim()) { continue }
  $want, $name = $line.Trim() -split "\s+", 2
  $file = Join-Path $Pkg $name
  if (-not (Test-Path $file)) { throw "В пакете нет файла $name" }
  $have = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
  if ($have -ne $want.ToLower()) { throw "Файл $name повреждён при копировании - скопируйте папку пакета на флешку заново" }
  Write-Host "  ${name}: OK"
}

Set-Location $App
# через cmd: в Windows PowerShell 5.1 перенаправление ошибок программы при ErrorActionPreference=Stop обрывает скрипт
cmd /c "docker info >nul 2>&1"
if ($LASTEXITCODE -ne 0) { throw "Docker не запущен: откройте Docker Desktop, дождитесь значка 'Engine running' и повторите" }

Write-Host "== 2/6 резервная копия базы перед обновлением"
$running = @(cmd /c "docker compose ps --status running --services 2>nul")
if ((Test-Path ".env") -and ($running -contains "db")) {
  New-Item -ItemType Directory -Force -Path "backups" | Out-Null
  $stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
  if ($running -contains "backup") {
    Run-Docker compose exec -T backup pg_dump -Fc -f "/backups/tolstoy-$stamp.dump"
  } else {
    Run-Docker compose exec -T db sh -c 'pg_dump -U $POSTGRES_USER -d $POSTGRES_DB -Fc -f /tmp/pre-update.dump'
    Run-Docker compose cp "db:/tmp/pre-update.dump" "backups/tolstoy-$stamp.dump"
  }
  $dump = Get-Item "backups\tolstoy-$stamp.dump"
  if ($dump.Length -lt 1000) { throw "Резервная копия пустая - обновление остановлено" }
  Write-Host ("  копия сохранена: backups\tolstoy-$stamp.dump ({0:N1} МБ)" -f ($dump.Length / 1MB))
} else {
  Write-Host "  база не запущена - копия пропущена (первая установка?)"
}

Write-Host "== 3/6 загрузка образов (несколько минут)"
$images = Join-Path $Pkg "images.tar"
if (-not (Test-Path $images)) { $images = Join-Path $Pkg "images.tar.gz" }
Run-Docker load -i $images

Write-Host "== 4/6 обновление кода в $App"
& tar -xzf (Join-Path $Pkg "code.tar.gz") -C $App
if ($LASTEXITCODE -ne 0) { throw "Не удалось распаковать code.tar.gz" }
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  throw "Создан .env из образца: впишите пароли (DB_PASSWORD, ADMIN_PASSWORD, SECRET_KEY) и запустите скрипт снова"
}

$dc = @("compose", "-f", "docker-compose.yml", "-f", "deploy/compose.offline.yml")
if ($Https) { $dc += @("--profile", "prod") }
$models = Join-Path $Pkg "stanza_models.tar.gz"
if (Test-Path $models) {
  Write-Host "== 5/6 модели Stanza"
  Run-Docker @dc run --rm --no-deps -v "${Pkg}:/pkg:ro" --entrypoint tar worker xzf /pkg/stanza_models.tar.gz -C /models/stanza
} else {
  Write-Host "== 5/6 моделей в пакете нет - остаются те, что уже на сервере"
}

Write-Host "== 6/6 перезапуск"
Run-Docker @dc up -d --no-build --remove-orphans
Start-Sleep -Seconds 8
Run-Docker @dc ps
Write-Host "Готово. Версия - внизу бокового меню сайта; журнал: docker compose logs --tail 50 web"
