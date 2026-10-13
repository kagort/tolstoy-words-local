from commons.config.db_config import engine_natural
from database.model import Base

Base.metadata.create_all(engine_natural)