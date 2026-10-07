from os import environ

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Читаем файл .env из корня проекта и кладём его значения в переменные окружения.
# Сам файл .env в Git не попадает (см. .gitignore), образец — .env.example.
load_dotenv()

# Логин и пароль обязательны: если их нет, программа сразу упадёт с KeyError,
# а не подключится молча с чужим паролем «по умолчанию».
DB_USER = environ["DB_USER"]
DB_PASSWORD = environ["DB_PASSWORD"]

# Остальное можно не задавать — тогда берутся безопасные значения для локальной работы.
DB_HOST = environ.get("DB_HOST", "localhost")
DB_PORT = environ.get("DB_PORT", "5432")
DB_NAME_GENERIC = environ.get("DB_NAME_GENERIC", "generic")
DB_NAME_NATURAL = environ.get("DB_NAME_NATURAL", "natural")

engine_generic = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_GENERIC}")
engine_natural = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_NATURAL}")
