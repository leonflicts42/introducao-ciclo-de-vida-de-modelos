import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import joblib
import pandas as pd
from flask import Flask, jsonify, request

sys.path.append(os.path.dirname(__file__))
from preprocessing import HeartDiseasePreprocessor

# -------------------------------------------------
# Utilidades de caminho
# -------------------------------------------------

def _abs_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, *parts))

MODEL_ENV_VAR = "MODEL_PATH"
DEFAULT_MODEL_PATH = _abs_path("..", "..", "models", "best_random_forest.joblib")

LOG_ENV_VAR = "REQUEST_LOG_PATH"
DEFAULT_LOG_PATH = _abs_path("requests.log")

app = Flask(__name__)
model = None
model_path_loaded = None
preprocessor: Optional[HeartDiseasePreprocessor] = None

# -------------------------------------------------
# Configuração de logging estruturado
# -------------------------------------------------
logger = logging.getLogger("request_logger")
logger.setLevel(logging.INFO)
_log_path = os.getenv(LOG_ENV_VAR, DEFAULT_LOG_PATH)
if not logger.handlers:
    fh = logging.FileHandler(_log_path)
    fh.setLevel(logging.INFO)    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def log_request(
    timestamp: str,
    request_id: str,
    request_payload: str,
    response_payload: Dict[str, Any],
    status_code: int,
    latency_ms: float
) -> None:
    """Grava uma linha JSON com os campos especificados."""
    try:
        record = {
            "timestamp": timestamp,
            "request_id": request_id,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
        }
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        # Evitar que falha de logging quebre a requisição
        app.logger.error(f"Falha ao logar requisição: {e}")

# -------------------------------------------------
# Carregamento do modelo
# -------------------------------------------------

def load_model() -> Tuple[Any, str]:
    path = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_PATH)
    mdl = joblib.load(path)
    return mdl, path

# -------------------------------------------------
# Normalização de payload de entrada
# -------------------------------------------------

def normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "instances" in payload and isinstance(payload["instances"], list):
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Formato de JSON inválido. Envie um objeto, lista de objetos ou {\"instances\": [...]}.")

# -------------------------------------------------
# Endpoints
# -------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    ts = datetime.utcnow().isoformat() + "Z"
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    resp_dict = {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path_loaded,
        "time": ts,
    }
    status_code = 200
    latency_ms = (time.perf_counter() - start) * 1000.0
    # Logar também requisições de health para demonstrar cobertura total
    log_request(
        timestamp=ts,
        request_id=request_id,
        request_payload=request.get_data(as_text=True) or "",
        response_payload=resp_dict,
        status_code=status_code,
        latency_ms=latency_ms,
    )
    return jsonify(resp_dict), status_code

@app.route("/heart-disease-predict", methods=["POST"])
def heart_disease_predict():
    ts = datetime.utcnow().isoformat() + "Z"
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    raw_payload = request.get_data(as_text=True) or ""

    status_code = 200
    response_obj: Dict[str, Any] = {}

    try:
        if model is None:
            response_obj = {"error": "Modelo não carregado."}
            status_code = 503
            return jsonify(response_obj), status_code
        if not request.is_json:
            response_obj = {"error": "Content-Type deve ser application/json"}
            status_code = 400
            return jsonify(response_obj), status_code

        rows = normalize_payload(request.get_json(silent=True))
        if not rows:
            response_obj = {"error": "Payload vazio."}
            status_code = 400
            return jsonify(response_obj), status_code

        df = pd.DataFrame(rows)
        if preprocessor is None:
            response_obj = {"error": "Pré-processador não inicializado."}
            status_code = 500
            return jsonify(response_obj), status_code

        df_aligned = preprocessor.transform(df)
        preds = model.predict(df_aligned)

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_aligned)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    proba = proba[:, 1]
                proba = proba.tolist()
            except Exception:
                proba = None

        response_obj = {"predictions": preds.tolist()}
        if proba is not None:
            response_obj["probabilities"] = proba
        return jsonify(response_obj), status_code

    except ValueError as ve:
        response_obj = {"error": str(ve)}
        status_code = 400
        return jsonify(response_obj), status_code
    except Exception as e:
        response_obj = {"error": f"Falha ao processar pedido: {e}"}
        status_code = 500
        return jsonify(response_obj), status_code
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        log_request(
            timestamp=ts,
            request_id=request_id,
            request_payload=raw_payload,
            response_payload=response_obj,
            status_code=status_code,
            latency_ms=latency_ms,
        )

# -------------------------------------------------
# Rotina de inicialização
# -------------------------------------------------

def _startup_load():
    global model, model_path_loaded, preprocessor
    try:
        model, model_path_loaded = load_model()
        preprocessor = HeartDiseasePreprocessor(model)
    except Exception as e:
        model = None
        model_path_loaded = f"(erro ao carregar: {e})"

_startup_load()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.logger.info(f"Iniciando servidor Flask na porta {port}; log em {_log_path}")
    app.run(host="0.0.0.0", port=port, debug=debug)
