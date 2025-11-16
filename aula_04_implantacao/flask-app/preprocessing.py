from typing import Optional, List

import numpy as np
import pandas as pd


class HeartDiseasePreprocessor:
    def __init__(self, model) -> None:
        self.model = model
        self.expected_feature_order: Optional[List[str]] = None
        self._init_expected_order()

    def _init_expected_order(self) -> None:
        """
        Tenta extrair a ordem esperada de features do modelo treinado.
        """
        try:
            if self.model is None:
                return
            if hasattr(self.model, "feature_names_in_"):
                self.expected_feature_order = list(getattr(self.model, "feature_names_in_"))
            elif hasattr(self.model, "named_steps") and "model" in getattr(self.model, "named_steps"):
                inner = self.model.named_steps["model"]
                if hasattr(inner, "feature_names_in_"):
                    self.expected_feature_order = list(getattr(inner, "feature_names_in_"))
        except Exception:
            # Be permissive; alignment will be skipped
            self.expected_feature_order = None

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduz o processo de Engenharia de Features visto na Aula 3:
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

        for c in out.columns:
            if pd.api.types.is_object_dtype(out[c]):
                try:
                    out[c] = pd.to_numeric(out[c])
                except Exception:
                    pass

        if "target" in out.columns:
            out = out.drop(columns=["target"])

        return out

    def apply_raw_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduz o tratamento de encoding visto na Aula 2:
        - Remove coluna 'dataset' (se existir)
        - Converte colunas categóricas (sex, cp, restecg, slope, thal) para strings canônicas
          quando vierem como códigos numéricos
        - Aplica pd.get_dummies(columns=[...], drop_first=True)
        """
        out = df.copy()

        # Remover 'dataset' para alinhar com Aula 2
        if 'dataset' in out.columns:
            out = out.drop(columns=['dataset'])

        # Mapear códigos numéricos -> strings canônicas usadas no CSV pré-processado
        def map_col(col: str, mapping: dict):
            if col in out.columns:
                series = out[col]
                # Converter boolean/string numérico para int quando apropriado
                if series.dtype == bool:
                    numeric = series.astype(int)
                else:
                    try:
                        numeric = pd.to_numeric(series)
                    except Exception:
                        numeric = series

                if pd.api.types.is_numeric_dtype(numeric):
                    vals = numeric.astype(int).values
                    out[col] = [mapping.get(int(v), mapping.get(v, series.iloc[i] if i < len(series) else v))
                                for i, v in enumerate(vals)]
                else:
                    # Já string, normalizar por chave inversa se necessário
                    inv = {v: v for v in mapping.values()}
                    out[col] = [inv.get(str(x), str(x)) for x in series]

        sex_map = {0: 'Female', 1: 'Male'}
        cp_map = {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal', 3: 'asymptomatic'}
        rest_map = {0: 'normal', 1: 'st-t abnormality', 2: 'left ventricular hypertrophy'}
        slope_map = {0: 'upsloping', 1: 'flat', 2: 'downsloping'}
        thal_map = {0: 'normal', 1: 'fixed defect', 2: 'reversable defect'}

        map_col('sex', sex_map)
        map_col('cp', cp_map)
        map_col('restecg', rest_map)
        map_col('slope', slope_map)
        map_col('thal', thal_map)

        # Aplicar get_dummies exatamente como na Aula 2 (drop_first=True)
        categorical_cols = [c for c in ['sex', 'cp', 'restecg', 'slope', 'thal'] if c in out.columns]
        if categorical_cols:
            out = pd.get_dummies(out, columns=categorical_cols, drop_first=True)

        return out

    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que as colunas estejam na ordem esperada pelo modelo
        """
        if self.expected_feature_order is None:
            return df
        aligned = df.copy()
        for col in self.expected_feature_order:
            if col not in aligned.columns:
                aligned[col] = 0
        aligned = aligned[self.expected_feature_order]
        for c in aligned.columns:
            if aligned[c].dtype == bool:
                aligned[c] = aligned[c].astype(int)
        return aligned

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o pipeline completo de pré-processamento:
        1. Raw OHE
        2. Engenharia de Features
        3. Garantir que as colunas estejam na ordem esperada pelo modelo
        """        
        out = self.apply_raw_categorical_encoding(df)
        out = self.apply_feature_engineering(out)
        out = self.align_features(out)
        return out
