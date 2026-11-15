# Восстановление базы из резервной копии на Windows (в папке проекта):
#   powershell -ExecutionPolicy Bypass -File deploy\restore.ps1 backups\tolstoy-2026-11-15_0300.dump
# ВНИМАНИЕ: текущее содержимое базы будет заменено. Для Linux — deploy/restore.sh.
param([Parameter(Mandatory = $true)][string]$File)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Run-Docker { & docker.exe @args; if ($LASTEXITCODE -ne 0) { throw "Ошибка команды: docker $($args -join ' ')" } }
if (-not (Test-Path $File)) { throw "Нет файла копии: $File" }
Run-Docker compose stop web worker
Run-Docker compose cp $File "db:/tmp/restore.dump"
Run-Docker compose exec -T db sh -c 'pg_restore -U $POSTGRES_USER -d $POSTGRES_DB --clean --if-exists /tmp/restore.dump'
Run-Docker compose start web worker
Write-Host "Восстановлено из $File"
