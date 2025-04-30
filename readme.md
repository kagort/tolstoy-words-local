# Tolstoy Words Local

## Инструкция для запуска

### 1. Создание и активация виртуального окружения

```bash
python3 -m venv .venv

source .venv/bin/activate
or
.venv\Scripts\activate
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Запуск web сервиса

```bash
.venv\Scripts\flask.exe --app web-project\app\app.py run
or
.venv\Scripts\flask.exe --app web-project\app\app.py --debug run
```

