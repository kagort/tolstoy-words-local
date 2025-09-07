from os import environ
from sqlalchemy import create_engine

DB_USER = environ.get('DB_USER', 'admin')
DB_PASSWORD = environ.get('DB_PASSWORD', 'password')
DB_HOST = environ.get('DB_HOST', 'pgadmin.quotverba.ru')
DB_PORT = environ.get('DB_PORT', '5432')
DB_NAME_GENERIC = environ.get('DB_NAME', 'generic')
DB_NAME_NATURAL = environ.get('DB_NAME', 'natural')

engine_generic = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_GENERIC}')
engine_natural = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_NATURAL}')