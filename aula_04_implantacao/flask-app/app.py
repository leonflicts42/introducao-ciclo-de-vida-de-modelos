import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple
import joblib
import pandas as pd
from flask import Flask, jsonify, request
sys.path.append(os.path.dirname(__file__))

from preprocessing import HeartDiseasePreprocessor

def _abs_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, *parts))


MODEL_ENV_VAR = "MODEL_PATH"
# When inside aula_04_implantacao/flask-app/, go two levels up to repo root
DEFAULT_MODEL_PATH = _abs_path("..", "..", "models", "best_random_forest.joblib")

app = Flask(__name__)
model = None
model_path_loaded = None
preprocessor: HeartDiseasePreprocessor | None = None


def load_model() -> Tuple[Any, str]:
    """Load the trained model pipeline from disk.

    Priority:
    - MODEL_PATH env var
    - default path under ../../models/best_random_forest.joblib
    """
    path = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_PATH)
    mdl = joblib.load(path)
    return mdl, path

def normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    """Accepts multiple JSON shapes and returns a list of row dicts.

    Supported shapes:
    - {"instances": [ {feature: value}, ... ]}
    - [ {feature: value}, ... ]
    - {feature: value} (single row)
    """
    if isinstance(payload, dict) and "instances" in payload and isinstance(payload["instances"], list):
        return payload["instances"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Formato de JSON inválido. Envie um objeto, lista de objetos ou {\"instances\": [...]}.")


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": model_path_loaded,
        "time": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/heart-disease-predict", methods=["POST"])
def heart_disease_predict():
    if model is None:
        return jsonify({"error": "Modelo não carregado."}), 503
    if not request.is_json:
        return jsonify({"error": "Content-Type deve ser application/json"}), 400

    try:
        rows = normalize_payload(request.get_json(silent=True))
        if not rows:
            return jsonify({"error": "Payload vazio."}), 400

        df = pd.DataFrame(rows)
        # Full preprocessing: raw OHE -> feature engineering -> alignment
        if preprocessor is None:
            return jsonify({"error": "Pré-processador não inicializado."}), 500
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

        resp = {
            "predictions": preds.tolist(),
        }
        if proba is not None:
            resp["probabilities"] = proba

        return jsonify(resp)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Falha ao processar pedido: {e}"}), 500


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
    app.run(host="0.0.0.0", port=port, debug=debug)
