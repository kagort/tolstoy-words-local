import model
from commons.config.db_config import engine_generic

model.Base.metadata.create_all(engine_generic)