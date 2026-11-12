# Пакет обновления для сервера без интернета — Windows (PowerShell), компьютер С интернетом и Docker Desktop.
# В папке проекта:   powershell -ExecutionPolicy Bypass -File deploy\offline-pack.ps1 [-Out offline-update] [-WithModels] [-AppOnly]
# Папку пакета потом целиком копируют на флешку; на сервере — offline-install.sh из пакета (docs/DEPLOY.md §2.8).
param([string]$Out = "offline-update", [switch]$WithModels, [switch]$AppOnly)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run($cmd) { Write-Host "> $cmd"; cmd /c $cmd; if ($LASTEXITCODE -ne 0) { throw "Ошибка: $cmd" } }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$OutFull = (Resolve-Path $Out).Path
$version = (Select-String -Path app\__init__.py -Pattern '__version__ = "(.*)"').Matches[0].Groups[1].Value
Write-Host "== Quotverba $version: пакет обновления в $OutFull"

Write-Host "== 1/4 сборка образа приложения (нужен интернет)"
Run "docker compose build web"
$images = "quotverba:latest"
if (-not $AppOnly) {
  Write-Host "== 2/4 образы базы данных и HTTPS-прокси"
  Run "docker pull postgres:16"; Run "docker pull caddy:2"
  $images = "quotverba:latest postgres:16 caddy:2"
}
Write-Host "== 3/4 сохранение образов ($images) — несколько минут"
Run "docker save -o `"$OutFull\images.tar`" $images"
Run "git archive --format=tar.gz -o `"$OutFull\code.tar.gz`" HEAD"
Copy-Item deploy\offline-install.sh $OutFull -Force

$files = @("images.tar", "code.tar.gz", "offline-install.sh")
if ($WithModels) {
  Write-Host "== 4/4 модели Stanza из тома Docker"
  $vol = (docker volume ls -q --filter label=com.docker.compose.volume=stanza_models | Select-Object -First 1)
  if (-not $vol) { throw "Том с моделями не найден: запустите приложение здесь, дождитесь «Загрузка моделей Stanza», повторите с -WithModels" }
  Run "docker run --rm -v ${vol}:/m:ro -v `"${OutFull}:/out`" --entrypoint tar quotverba:latest czf /out/stanza_models.tar.gz -C /m ."
  $files += "stanza_models.tar.gz"
}
$commit = (git rev-parse --short HEAD)
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$OutFull\ВЕРСИЯ.txt", "Quotverba $version, пакет от $(Get-Date -Format 'yyyy-MM-dd HH:mm'), коммит $commit`nУстановка на сервере: ./offline-install.sh /путь/к/проекту   (см. docs/DEPLOY.md §2.8)`n", $utf8)
# контрольные суммы в формате sha256sum (переводы строк LF) — их проверяет offline-install.sh на сервере
$sums = ($files | ForEach-Object { "$((Get-FileHash -Algorithm SHA256 (Join-Path $OutFull $_)).Hash.ToLower())  $_" }) -join "`n"
[IO.File]::WriteAllText("$OutFull\SHA256SUMS", "$sums`n", $utf8)
Write-Host "Готово. Скопируйте папку $OutFull на флешку целиком."
