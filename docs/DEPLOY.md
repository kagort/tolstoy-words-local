# Развёртывание: локально (Windows) и на VPS

Приложение состоит из четырёх частей, и все они запускаются одной командой через Docker Compose:

| Контейнер | Что делает |
|---|---|
| `db` | PostgreSQL 16 — база данных; данные лежат в томе `pgdata` и переживают перезапуск |
| `web` | Веб-интерфейс (FastAPI); при старте сам обновляет схему БД |
| `worker` | Фоновый обработчик: разбор текстов, пересчёт словарей |
| `caddy` | Только на VPS: HTTPS-сертификат Let's Encrypt и проксирование на `web` |

---

## Часть 1. Локально на Windows

### 1.1. Что нужно установить один раз
- **Docker Desktop** (у вас уже есть). Запустите его и дождитесь зелёного статуса «Engine running».
- **Git** и **VS Code** (уже есть).

### 1.2. Получить код
```powershell
cd C:\Projects\tolstoy-words-local
git checkout master
git pull origin master
```

### 1.3. Создать файл настроек `.env`
```powershell
Copy-Item .env.example .env
code .env
```
Заполните три значения (придумайте новые, не используйте старые пароли из истории репозитория):
- `DB_PASSWORD` — пароль базы данных;
- `ADMIN_PASSWORD` — пароль для входа в веб-интерфейс (логин — `admin`);
- `SECRET_KEY` — любая длинная случайная строка (например, 40 случайных букв и цифр).

⚠️ **Не используйте символ `$` в значениях `.env`**: Docker Compose считает его началом имени переменной и молча вырезает часть пароля (признак — предупреждение `The "…" variable is not set`). Безопасный пароль из букв и цифр в PowerShell:
```powershell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
```
Проверить файл: `docker compose config --quiet` — если ничего не вывелось, всё в порядке.

Остальное можно не трогать. `NLP_BACKEND=stanza` — точный парсер (образ соберётся дольше и будет весить ~2 ГБ). Для быстрой пробы можно поставить `NLP_BACKEND=natasha`.

### 1.4. Запустить
```powershell
docker compose up -d --build
```
Первый запуск со Stanza занимает 5–15 минут: скачиваются Python-пакеты, PyTorch и русская модель. Последующие запуски — секунды.

Проверить, что всё поднялось:
```powershell
docker compose ps
```
Все три сервиса (`db`, `web`, `worker`) должны быть в состоянии `running` (`db` — `healthy`).

Откройте в браузере **http://localhost:8000**, войдите (`admin` + ваш `ADMIN_PASSWORD`).

### 1.5. Проверка парсера
```powershell
docker compose run --rm worker python -m app.cli check-nlp
```
Должна напечататься таблица разбора (CoNLL-U) предложения про сирень и список связей. Первая строка — `Парсер: stanza-…` или `natasha-…`.
Если Stanza выдаёт ошибку — пришлите её текст; временно поставьте в `.env` `NLP_BACKEND=natasha` и выполните `docker compose up -d --build`.

### 1.6. Демо-данные и массовая загрузка корпуса
Демо (3 текста из `data/raw`, пример текста нейросети и словарь запахов):
```powershell
docker compose run --rm worker python -m app.cli seed-demo
```
Весь корпус `data/raw` (71 текст) с метаданными из `data/processed/dictexts.csv`:
```powershell
docker compose run --rm worker python -m app.cli import-dir data/raw --meta data/processed/dictexts.csv --corpus "Литература"
docker compose run --rm worker python -m app.cli import-dir data/generated --corpus "Тексты нейросетей" --type generated
```
Тексты ставятся в очередь; прогресс виден на странице «Задачи». Разбор 101 МБ на обычном ноутбуке займёт много часов (Stanza) — компьютер можно оставить на ночь; после перезапуска обработчик продолжит с незавершённых задач.

### 1.7. Повседневные команды
| Действие | Команда |
|---|---|
| Остановить | `docker compose stop` |
| Запустить снова | `docker compose start` |
| Обновить после `git pull` | `docker compose up -d --build` |
| Логи веб-интерфейса / обработчика | `docker compose logs -f web` / `docker compose logs -f worker` |
| Подключиться к БД из DBeaver/pgAdmin | хост `localhost`, порт `5432`, база `tolstoy`, пользователь `postgres`, пароль из `.env` |
| Удалить всё вместе с базой (осторожно!) | `docker compose down -v` |

---

## Часть 2. VPS с доступом из интернета

### 2.1. Какой сервер нужен
- Ubuntu 24.04 LTS (или 22.04).
- Для Stanza: **4 vCPU, 8 ГБ RAM**, 40+ ГБ SSD (минимум 4 ГБ RAM). Для Natasha хватит 2 ГБ RAM.
- Доменное имя (например, `tolstoy.example.ru`) с A-записью на IP сервера — нужно для HTTPS.

### 2.2. Первичная настройка сервера (один раз)
Подключитесь по SSH (из PowerShell: `ssh root@IP_СЕРВЕРА`) и выполните:
```bash
# обновления и файрвол: открыты только SSH, HTTP и HTTPS
apt update && apt upgrade -y
apt install -y ufw fail2ban git
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable

# Docker
curl -fsSL https://get.docker.com | sh

# отдельный пользователь вместо root
adduser tw && usermod -aG docker,sudo tw
```
Рекомендуется вход по SSH-ключу вместо пароля: на Windows `ssh-keygen`, затем `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh tw@IP "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"`. После проверки входа по ключу отключите вход по паролю (`PasswordAuthentication no` в `/etc/ssh/sshd_config`, затем `systemctl restart ssh`).

### 2.3. Установка приложения
```bash
su - tw
sudo mkdir -p /opt/quotverba && sudo chown tw /opt/quotverba
git clone https://github.com/kagort/tolstoy-words-local.git /opt/quotverba
cd /opt/quotverba
cp .env.example .env
nano .env
```
В `.env` на сервере обязательно:
- **новые** длинные `DB_PASSWORD`, `ADMIN_PASSWORD`, `SECRET_KEY` (сгенерировать: `openssl rand -base64 32`);
- раскомментировать и заполнить `DOMAIN=ваш.домен`, `ACME_EMAIL=ваша@почта`, `HTTPS_ONLY=1`.

Запуск с HTTPS:
```bash
docker compose --profile prod up -d --build
docker compose ps
```
Через минуту откройте `https://ваш.домен` — Caddy сам получит сертификат. Веб-контейнер слушает только `127.0.0.1:8000`, наружу смотрит только Caddy (порты 80/443). PostgreSQL наружу **не открыт**.

### 2.4. Доступ к базе с вашего компьютера (DBeaver/pgAdmin)
Через SSH-туннель — безопасно и без открытия порта:
```powershell
ssh -L 5433:localhost:5432 tw@IP_СЕРВЕРА
```
Пока окно открыто, в DBeaver подключайтесь к `localhost:5433`, база `tolstoy`, пользователь `postgres`, пароль из `.env` сервера.

### 2.5. Резервные копии
```bash
cd /opt/quotverba
./deploy/backup.sh                 # копия в backups/, старше 14 дней удаляются
crontab -e                         # ежедневно в 03:15:
# 15 3 * * * cd /opt/quotverba && ./deploy/backup.sh >> backups/backup.log 2>&1
```
Скачать копию к себе: `scp tw@IP:/opt/quotverba/backups/ИМЯ.dump .`
Восстановление: `./deploy/restore.sh backups/ИМЯ.dump`.
Раз в месяц проверяйте восстановление на локальной машине — копия, которую ни разу не восстанавливали, не считается копией.

### 2.6. Перенос базы с компьютера на VPS
Удобно разобрать большой корпус на мощном компьютере, а потом перенести готовую базу:
```powershell
# на Windows, в папке проекта
docker compose exec -T db pg_dump -U postgres -d tolstoy -Fc > quotverba.dump
scp quotverba.dump tw@IP:/opt/quotverba/backups/
```
```bash
# на VPS
cd /opt/quotverba && ./deploy/restore.sh backups/quotverba.dump
```

### 2.7. Обновление на VPS
```bash
cd /opt/quotverba
./deploy/backup.sh
git pull
docker compose --profile prod up -d --build
```

---

## Часть 3. Запуск без Docker (для разработки)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
# для Stanza дополнительно:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-stanza.txt
```
Нужен PostgreSQL (можно только контейнер БД: `docker compose up -d db`). В `.env`: `DB_HOST=localhost`.
```powershell
alembic upgrade head
uvicorn app.main:app --reload          # окно 1: веб-интерфейс на http://localhost:8000
python -m app.worker                    # окно 2: обработчик задач
```
Тесты: `pytest` (модульные); со сквозным сценарием — задайте `TEST_DATABASE_URL` на отдельную пустую базу (она очищается тестом!).

## Если что-то не работает

| Симптом | Что сделать |
|---|---|
| `DB_PASSWORD … Задайте DB_PASSWORD в файле .env` | Не создан `.env` или в нём нет `DB_PASSWORD` (шаг 1.3) |
| Страница входа: «Пароль администратора не задан» | Задайте `ADMIN_PASSWORD` в `.env`, затем `docker compose up -d` |
| Тексты висят «в очереди» | Не запущен обработчик: `docker compose ps`, `docker compose logs worker` |
| `exec /entrypoint.sh: no such file or directory` | Скрипт получил концы строк Windows (CRLF). Выполните `git rm --cached -r . ; git reset --hard` (после `.gitattributes` файлы перезапишутся с LF) и пересоберите |
| `failed to connect to the docker API … dockerDesktopLinuxEngine` | Не запущен Docker Desktop: запустите его и дождитесь «Engine running»; проверка — `docker version` (должен быть раздел Server) |
| `The "K" variable is not set` (или другая буква) | В `.env` есть символ `$` — замените такие пароли (см. шаг 1.3). Если база уже создавалась со «вырезанным» паролем: `docker compose down -v` (удалит пустую базу) и запустить заново |
| `ports are not available: exposing port TCP 127.0.0.1:5432 … access permissions` | Порт 5432 занят (установлен PostgreSQL для Windows) или зарезервирован Windows/Hyper-V (`netsh interface ipv4 show excludedportrange protocol=tcp`). В `.env` добавьте `DB_PUBLISH_PORT=15432` и снова `docker compose up -d`. Порт нужен только для DBeaver/pgAdmin — приложение внутри Docker работает с базой напрямую |
| Порт 5432 или 8000 занят | В `.env`: `DB_PUBLISH_PORT=5433` или `WEB_PORT=8080` |
| Caddy не получает сертификат | Проверьте, что A-запись домена указывает на IP сервера и порты 80/443 открыты (`ufw status`) |
