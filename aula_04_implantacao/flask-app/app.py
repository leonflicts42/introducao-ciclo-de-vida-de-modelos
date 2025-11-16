import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


def _abs_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, *parts))


MODEL_ENV_VAR = "MODEL_PATH"
# When inside aula_04_implantacao/flask-app/, go two levels up to repo root
DEFAULT_MODEL_PATH = _abs_path("..", "..", "models", "best_random_forest.joblib")

app = Flask(__name__)
model = None
model_path_loaded = None
expected_feature_order = None


def load_model() -> Tuple[Any, str]:
    """Load the trained model pipeline from disk.

    Priority:
    - MODEL_PATH env var
    - default path under ../../models/best_random_forest.joblib
    """
    path = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_PATH)
    mdl = joblib.load(path)
    return mdl, path


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates Aula 3 engineered features on input DataFrame when possible.

    If engineered columns already exist, they are preserved; otherwise computed
    using available base columns. Missing base columns are ignored gracefully.
    """
    out = df.copy()
    eps = 1

    if "age" in out.columns:
        if "age_squared" not in out.columns:
            out["age_squared"] = out["age"] ** 2
        if "age_decade" not in out.columns:
            out["age_decade"] = (out["age"] // 10).astype(int)

    if "chol" in out.columns and "age" in out.columns and "cholesterol_to_age" not in out.columns:
        out["cholesterol_to_age"] = out["chol"] / (out["age"] + eps)

    if "thalch" in out.columns and "age" in out.columns and "max_hr_pct" not in out.columns:
        predicted_max_hr = (220 - out["age"]).clip(lower=1)
        out["max_hr_pct"] = out["thalch"] / (predicted_max_hr + eps)

    if "trestbps" in out.columns and "chol" in out.columns and "bp_chol_ratio" not in out.columns:
        out["bp_chol_ratio"] = out["trestbps"] / (out["chol"] + 1)

    if "fbs" in out.columns and "fbs_flag" not in out.columns:
        out["fbs_flag"] = out["fbs"].astype(int)

    if "exang" in out.columns and "exang_flag" not in out.columns:
        out["exang_flag"] = out["exang"].astype(int)

    if "thalch" in out.columns and "trestbps" in out.columns and "stress_index" not in out.columns:
        out["stress_index"] = out["thalch"] / (out["trestbps"] + eps)

    if "age" in out.columns and "oldpeak" in out.columns and "risk_interaction" not in out.columns:
        out["risk_interaction"] = out["age"] * out["oldpeak"]

    if "oldpeak" in out.columns and "high_st_depression_flag" not in out.columns:
        out["high_st_depression_flag"] = (out["oldpeak"] > 1.0).astype(int)

    # Ensure numeric types where possible
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            try:
                out[c] = pd.to_numeric(out[c])
            except Exception:
                pass

    # Drop target if user sent it
    if "target" in out.columns:
        out = out.drop(columns=["target"])

    return out


def apply_raw_categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw categorical columns (cp, restecg, slope, thal, sex) to the
    one-hot schema found in the preprocessed Aula 2 dataset if those one-hot
    columns are not already present.

    Expected one-hot columns (subset observed in dataset header):
    - sex_Male (bool)
    - cp_atypical angina, cp_non-anginal, cp_typical angina (cp asymptomatic omitted in header; handle if present)
    - restecg_normal, restecg_st-t abnormality (LVH column absent in header: restecg_left ventricular hypertrophy)
    - slope_flat, slope_upsloping (downsloping omitted in header; handle if present)
    - thal_normal, thal_reversable defect (fixed defect omitted in header; handle if present)

    Any missing expected one-hot columns are created with 0/False values so the model's
    feature alignment works. Original raw columns are dropped after encoding.
    """
    out = df.copy()

    # sex (assuming 1=Male, 0=Female or boolean True=Male)
    if "sex_Male" not in out.columns:
        if "sex" in out.columns:
            out["sex_Male"] = out["sex"].astype(int) == 1
            out.drop(columns=["sex"], inplace=True)
        elif "sex_Female" in out.columns and "sex_Male" not in out.columns:
            # already encoded differently; infer
            out["sex_Male"] = (~out["sex_Female"]).astype(bool)

    # Chest pain type cp (values: 0..3 or 1..4). Map heuristically.
    cp_ohe_cols_model = ["cp_atypical angina", "cp_non-anginal", "cp_typical angina"]
    if not any(c.startswith("cp_") for c in out.columns) and "cp" in out.columns:
        cp_series = out["cp"].astype(int)
        for col in cp_ohe_cols_model:
            out[col] = False
        cp_norm = np.where(cp_series >= 1, cp_series - 1, cp_series)  # normalize 1..4 -> 0..3
        mapping = {0: "cp_typical angina", 1: "cp_atypical angina", 2: "cp_non-anginal", 3: "cp_non-anginal"}  # fallback asymptomatic -> non-anginal
        for idx, val in enumerate(cp_norm):
            col_name = mapping.get(val)
            if col_name:
                out.loc[out.index[idx], col_name] = True
        out.drop(columns=["cp"], inplace=True)

    # Restecg (0=normal,1=st-t abnormality,2=lv hypertrophy)
    rest_cols_model = ["restecg_normal", "restecg_st-t abnormality"]
    if not any(c.startswith("restecg_") for c in out.columns) and "restecg" in out.columns:
        r = out["restecg"].astype(int)
        for col in rest_cols_model:
            out[col] = False
        mapping = {0: "restecg_normal", 1: "restecg_st-t abnormality", 2: "restecg_st-t abnormality"}  # fallback LVH -> abnormal
        for idx, val in enumerate(r):
            col_name = mapping.get(val)
            if col_name:
                out.loc[out.index[idx], col_name] = True
        out.drop(columns=["restecg"], inplace=True)

    # Slope (1=upsloping,2=flat,3=downsloping OR 0..2). Normalize similarly.
    slope_cols_model = ["slope_flat", "slope_upsloping"]
    if not any(c.startswith("slope_") for c in out.columns) and "slope" in out.columns:
        s = out["slope"].astype(int)
        s_norm = np.where(s > 0, s - 1, s)
        for col in slope_cols_model:
            out[col] = False
        mapping = {0: "slope_upsloping", 1: "slope_flat", 2: "slope_flat"}  # fallback downsloping -> flat
        for idx, val in enumerate(s_norm):
            col_name = mapping.get(val)
            if col_name:
                out.loc[out.index[idx], col_name] = True
        out.drop(columns=["slope"], inplace=True)

    # Thal (1=normal,2=fixed defect,3=reversable defect) or 0..2
    thal_cols_model = ["thal_normal", "thal_reversable defect"]
    if not any(c.startswith("thal_") for c in out.columns) and "thal" in out.columns:
        t = out["thal"].astype(int)
        t_norm = np.where(t > 0, t - 1, t)
        for col in thal_cols_model:
            out[col] = False
        mapping = {0: "thal_normal", 1: "thal_normal", 2: "thal_reversable defect"}  # fallback fixed defect -> normal
        for idx, val in enumerate(t_norm):
            col_name = mapping.get(val)
            if col_name:
                out.loc[out.index[idx], col_name] = True
        out.drop(columns=["thal"], inplace=True)

    # Ensure all columns observed in the preprocessed dataset exist (fill missing with False/0)
    expected_subset = [
        "sex_Male", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
        "restecg_normal", "restecg_st-t abnormality", "slope_flat", "slope_upsloping",
        "thal_normal", "thal_reversable defect"
    ]
    for col in expected_subset:
        if col not in out.columns:
            out[col] = False

    return out


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Align incoming DataFrame to the feature order and set expected by the
    trained model. Adds missing columns with 0, drops extras, and reorders.
    Falls back to original df if expected order not available.
    """
    global expected_feature_order
    if expected_feature_order is None and model is not None:
        # Try to extract once if model loaded
        if hasattr(model, "feature_names_in_"):
            expected_feature_order = list(getattr(model, "feature_names_in_"))
        elif hasattr(model, "named_steps") and "model" in getattr(model, "named_steps"):
            inner = model.named_steps["model"]
            if hasattr(inner, "feature_names_in_"):
                expected_feature_order = list(getattr(inner, "feature_names_in_"))
    if expected_feature_order is None:
        return df
    aligned = df.copy()
    # Add missing
    for col in expected_feature_order:
        if col not in aligned.columns:
            aligned[col] = 0
    # Keep only expected
    aligned = aligned[expected_feature_order]
    # Ensure numeric encoding for booleans
    for c in aligned.columns:
        if aligned[c].dtype == bool:
            aligned[c] = aligned[c].astype(int)
    return aligned


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
        # First create OHE categorical columns if user sent raw
        df_ohe = apply_raw_categorical_encoding(df)
        df_fe = apply_feature_engineering(df_ohe)
        df_aligned = align_features(df_fe)

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
    global model, model_path_loaded, expected_feature_order
    try:
        model, model_path_loaded = load_model()
        # Initialize expected feature order immediately if possible
        if model is not None:
            if hasattr(model, "feature_names_in_"):
                expected_feature_order = list(getattr(model, "feature_names_in_"))
            elif hasattr(model, "named_steps") and "model" in getattr(model, "named_steps"):
                inner = model.named_steps["model"]
                if hasattr(inner, "feature_names_in_"):
                    expected_feature_order = list(getattr(inner, "feature_names_in_"))
    except Exception as e:
        model = None
        model_path_loaded = f"(erro ao carregar: {e})"


_startup_load()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
