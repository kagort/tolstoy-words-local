from os import environ
from sqlalchemy import create_engine

DB_USER = environ.get('DB_USER', 'postgres')
DB_PASSWORD = environ.get('DB_PASSWORD', 'ouganda77')
DB_HOST = environ.get('DB_HOST', 'localhost')
DB_PORT = environ.get('DB_PORT', '5432')
DB_NAME_GENERIC = environ.get('DB_NAME', 'generic')
DB_NAME_NATURAL = environ.get('DB_NAME', 'tolstoy_words_csv')

engine_generic = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_GENERIC}')
engine_natural = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_NATURAL}')