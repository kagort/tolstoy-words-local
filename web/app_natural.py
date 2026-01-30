from api.api import TextAnalysisAPI
from commons.config.db_config import engine_natural

if __name__ == '__main__':

    api = TextAnalysisAPI(engine_natural)
    api.app.run(debug=True, host='0.0.0.0', port=5000)
