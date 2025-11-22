"""
Script de treinamento para CI/CD Pipeline.
Treina modelos (baseline ou otimizado) e loga no MLflow.
"""

import os
import sys
import warnings
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Importar transformers customizados
from preprocessing import CategoricalEncoder, FeatureEngineer, MissingValueImputer

warnings.filterwarnings('ignore')

# Configurações de hiperparâmetros
BASELINE_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42
}

OPTIMIZED_PARAMS = {
    'bootstrap': True,
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 15,
    'max_features': 2,
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 124,
    'oob_score': False,
    'random_state': 42,
    'verbose': 0,
    'warm_start': False
}


def load_and_prepare_data(data_path):
    """Carrega e prepara os dados para treinamento."""
    print(f"Carregando dados de: {data_path}")
    df = pd.read_csv(data_path)
    
    # Criar target binário
    if 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
    
    # Remover colunas de metadados
    columns_to_drop = ['id', 'dataset', 'num']
    cols_dropped = [col for col in columns_to_drop if col in df.columns]
    if cols_dropped:
        df = df.drop(columns=cols_dropped)
        print(f"Colunas removidas: {cols_dropped}")
    
    # Separar features e target
    X = df.drop(columns=['target'], errors='ignore')
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def create_pipeline(params, numeric_cols, categorical_cols):
    """Cria pipeline completo com pré-processamento e modelo."""
    pipeline = Pipeline([
        ('imputer', MissingValueImputer(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )),
        ('categorical_encoding', CategoricalEncoder()),
        ('feature_engineering', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**params))
    ])
    return pipeline


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Avalia o modelo e retorna métricas."""
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    return metrics


def train_model(model_type='baseline', data_path='../data/heart_disease_uci.csv',
                mlflow_experiment='heart-disease-cicd', min_accuracy=0.75):
    """
    Treina modelo e loga no MLflow.
    
    Args:
        model_type: 'baseline' ou 'optimized'
        data_path: caminho para o dataset
        mlflow_experiment: nome do experimento MLflow
        min_accuracy: acurácia mínima para registrar no Model Registry
    """
    print(f"\n{'='*60}")
    print(f"Iniciando treinamento: {model_type.upper()}")
    print(f"{'='*60}\n")
    
    # Carregar dados
    X, y = load_and_prepare_data(data_path)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    # Detectar colunas categóricas e numéricas
    all_cols = X_train.columns.tolist()
    categorical_cols = [c for c in ['sex', 'cp', 'restecg', 'slope', 'thal'] if c in all_cols]
    numeric_cols = [c for c in all_cols if c not in categorical_cols]
    
    # Selecionar hiperparâmetros
    if model_type == 'baseline':
        params = BASELINE_PARAMS
        run_name = "baseline_random_forest"
    else:
        params = OPTIMIZED_PARAMS
        run_name = "optimized_random_forest"
    
    print(f"Hiperparâmetros selecionados:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()
    
    # Criar e treinar pipeline
    pipeline = create_pipeline(params, numeric_cols, categorical_cols)
    print("Treinando pipeline completo...")
    pipeline.fit(X_train, y_train)
    
    # Avaliar modelo
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    print(f"\nMétricas de avaliação:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Logar no MLflow
    mlflow.set_experiment(mlflow_experiment)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Logar parâmetros
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param('model_type', model_type)
        
        # Logar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Criar signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        # Logar modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            code_paths=["preprocessing.py"],
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        print(f"\n✓ Modelo logado no MLflow")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model URI: {model_info.model_uri}")
        
        # Registrar no Model Registry se atingir acurácia mínima
        test_accuracy = metrics['test_accuracy']
        if test_accuracy >= min_accuracy:
            print(f"\n✓ Acurácia ({test_accuracy:.4f}) >= threshold ({min_accuracy})")
            print(f"  Registrando modelo no Model Registry...")
            
            try:
                model_name = "heart-disease-model"
                model_version = mlflow.register_model(
                    model_uri=model_info.model_uri,
                    name=model_name
                )
                
                print(f"  Modelo registrado: {model_name}")
                print(f"  Versão: {model_version.version}")
                
                # Promover para Production se for otimizado e melhor que baseline
                if model_type == 'optimized':
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    client.set_registered_model_alias(
                        model_name,
                        "Production",
                        model_version.version
                    )
                    print(f"  ✓ Alias 'Production' definido para versão {model_version.version}")
            except Exception as e:
                print(f"  ⚠ Erro ao registrar modelo: {e}")
        else:
            print(f"\n✗ Acurácia ({test_accuracy:.4f}) < threshold ({min_accuracy})")
            print(f"  Modelo NÃO será registrado no Model Registry")
    
    print(f"\n{'='*60}")
    print(f"Treinamento concluído!")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar modelo de Heart Disease')
    parser.add_argument(
        '--model-type',
        type=str,
        default='baseline',
        choices=['baseline', 'optimized'],
        help='Tipo de modelo: baseline ou optimized'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/heart_disease_uci.csv',
        help='Caminho para o dataset'
    )
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.75,
        help='Acurácia mínima para registrar no Model Registry'
    )
    
    args = parser.parse_args()
    
    metrics = train_model(
        model_type=args.model_type,
        data_path=args.data_path,
        min_accuracy=args.min_accuracy
    )
    
    # Retornar código de erro se acurácia for muito baixa
    if metrics['test_accuracy'] < 0.60:
        print("⚠ ERRO: Acurácia muito baixa! Pipeline falhando.")
        sys.exit(1)
