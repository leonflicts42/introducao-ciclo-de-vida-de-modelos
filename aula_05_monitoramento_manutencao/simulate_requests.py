"""
Script para simular requisições ao endpoint Flask e demonstrar data drift.

Este script envia 100 requisições ao endpoint de predição, simulando um cenário
de data drift onde as características dos dados em produção divergem dos dados
de treinamento.

Estratégia de drift simulado:
- Todas as 100 requisições apresentam drift progressivo
- Drift aumenta gradualmente de leve (início) para severo (final)
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, Any, List

# Configuração
ENDPOINT_URL = "http://localhost:5000/heart-disease-predict"
NUM_REQUESTS = 100
SEED = 42

np.random.seed(SEED)


def load_training_stats() -> Dict[str, Dict[str, float]]:
    """Carrega estatísticas dos dados de treino para simular distribuições similares."""
    df = pd.read_csv("../data/heart_disease_uci_preprocessed.csv")
    
    # Colunas numéricas originais (antes do one-hot encoding)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return stats


def generate_drifted_sample(stats: Dict[str, Dict[str, float]], drift_factor: float) -> Dict[str, Any]:
    """
    Gera uma amostra com drift simulado.
    
    drift_factor: 0.0 (sem drift) a 1.0 (drift máximo)
    
    Simulações de drift:
    - Idade: população mais velha (+10 anos em média)
    - Pressão arterial: valores mais elevados (+20 mmHg)
    - Colesterol: valores mais altos (+50 mg/dl)
    - Frequência cardíaca máxima: valores mais baixos (-20 bpm)
    - Mais casos de angina induzida por exercício
    - Distribuição de gênero desbalanceada (mais homens)
    """
    # Idade: drift para população mais velha
    age_shift = 10 * drift_factor
    age = int(np.clip(
        np.random.normal(stats['age']['mean'] + age_shift, stats['age']['std'] * 1.2),
        40, 80
    ))
    
    # Pressão arterial: drift para valores mais altos
    bp_shift = 20 * drift_factor
    trestbps = int(np.clip(
        np.random.normal(stats['trestbps']['mean'] + bp_shift, stats['trestbps']['std'] * 1.3),
        120, 200
    ))
    
    # Colesterol: drift para valores mais altos
    chol_shift = 50 * drift_factor
    chol = int(np.clip(
        np.random.normal(stats['chol']['mean'] + chol_shift, stats['chol']['std'] * 1.4),
        200, 600
    ))
    
    # Frequência cardíaca máxima: drift para valores mais baixos
    thalch_shift = -20 * drift_factor
    thalch = int(np.clip(
        np.random.normal(stats['thalch']['mean'] + thalch_shift, stats['thalch']['std'] * 1.1),
        80, 180
    ))
    
    # Oldpeak: drift para valores mais elevados (mais depressão ST)
    oldpeak_shift = 1.0 * drift_factor
    oldpeak = round(np.clip(
        np.random.normal(stats['oldpeak']['mean'] + oldpeak_shift, stats['oldpeak']['std'] * 1.5),
        0, 5
    ), 1)
    
    # CA: drift para mais vasos com fluoroscopia
    ca = int(np.clip(np.random.poisson(0.7 + drift_factor * 1.0), 0, 3))
    
    # Proporções com drift
    sex_prob = 0.68 + drift_factor * 0.15  # Mais homens
    exang_prob = 0.33 + drift_factor * 0.20  # Mais angina induzida
    fbs_prob = 0.15 + drift_factor * 0.10  # Mais diabetes
    
    sample = {
        'age': age,
        'sex': int(np.random.choice([0, 1], p=[1-sex_prob, sex_prob])),
        'cp': int(np.random.choice([0, 1, 2, 3], p=[0.40, 0.20, 0.30, 0.10])),  # Mais angina atípica
        'trestbps': trestbps,
        'chol': chol,
        'fbs': int(np.random.choice([0, 1], p=[1-fbs_prob, fbs_prob])),
        'restecg': int(np.random.choice([0, 1, 2], p=[0.40, 0.58, 0.02])),  # Mais anormalidades
        'thalch': thalch,
        'exang': int(np.random.choice([0, 1], p=[1-exang_prob, exang_prob])),
        'oldpeak': oldpeak,
        'slope': int(np.random.choice([0, 1, 2], p=[0.15, 0.40, 0.45])),  # Mais slope descendente
        'ca': ca,
        'thal': int(np.random.choice([0, 1, 2, 3], p=[0.02, 0.45, 0.45, 0.08]))  # Mais defeitos reversíveis
    }
    return sample


def send_request(payload: Dict[str, Any], request_num: int) -> Dict[str, Any]:
    """Envia requisição ao endpoint e retorna metadados."""
    try:
        start_time = time.time()
        response = requests.post(ENDPOINT_URL, json=payload, timeout=5)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'request_num': request_num,
            'status_code': response.status_code,
            'elapsed_ms': round(elapsed_ms, 2),
            'response': response.json() if response.ok else None,
            'error': None
        }
    except Exception as e:
        return {
            'request_num': request_num,
            'status_code': None,
            'elapsed_ms': None,
            'response': None,
            'error': str(e)
        }


def main():
    print("=" * 80)
    print("SIMULAÇÃO DE REQUISIÇÕES COM DATA DRIFT")
    print("=" * 80)
    print(f"\nEndpoint: {ENDPOINT_URL}")
    print(f"Total de requisições: {NUM_REQUESTS}")
    print(f"Estratégia: Drift progressivo de 0% (req. 1) até 100% (req. 100)\n")
    
    # Carrega estatísticas do treino
    print("Carregando estatísticas dos dados de treino...")
    stats = load_training_stats()
    
    results: List[Dict[str, Any]] = []
    
    print("\nIniciando envio de requisições...")
    print("-" * 80)
    
    for i in range(NUM_REQUESTS):
        # Drift progressivo de 0.0 (leve) a 1.0 (severo)
        drift_factor = i / (NUM_REQUESTS - 1)  # 0.0 a 1.0
        payload = generate_drifted_sample(stats, drift_factor)
        
        result = send_request(payload, i + 1)
        result['drift_factor'] = drift_factor
        result['payload'] = payload
        results.append(result)
        
        # Feedback visual
        status = "✓" if result['status_code'] == 200 else "✗"
        drift_pct = int(drift_factor * 100)
        print(f"[{i+1:3d}/100] {status} Drift:{drift_pct:3d}% | "
              f"Age: {payload['age']:2d} | BP: {payload['trestbps']:3d} | "
              f"Chol: {payload['chol']:3d} | Status: {result['status_code']}")
        
        # Pequeno delay para não sobrecarregar
        time.sleep(0.05)
    
    print("-" * 80)
    
    # Estatísticas finais
    successful = sum(1 for r in results if r['status_code'] == 200)
    failed = NUM_REQUESTS - successful
    avg_latency = np.mean([r['elapsed_ms'] for r in results if r['elapsed_ms']])
    
    print(f"\n{'RESUMO DA EXECUÇÃO':^80}")
    print("=" * 80)
    print(f"Requisições bem-sucedidas: {successful}/{NUM_REQUESTS} ({successful/NUM_REQUESTS*100:.1f}%)")
    print(f"Requisições com falha:     {failed}/{NUM_REQUESTS} ({failed/NUM_REQUESTS*100:.1f}%)")
    print(f"Latência média:            {avg_latency:.2f} ms")
    
    # Comparação entre início e fim (progressão do drift)
    first_20 = results[:20]
    last_20 = results[-20:]
    
    first_ages = [r['payload']['age'] for r in first_20]
    last_ages = [r['payload']['age'] for r in last_20]
    
    print(f"\n{'ANÁLISE DE DRIFT PROGRESSIVO':^80}")
    print("=" * 80)
    print(f"Idade média (primeiras 20):   {np.mean(first_ages):.1f} anos (drift baixo)")
    print(f"Idade média (últimas 20):     {np.mean(last_ages):.1f} anos (drift alto)")
    print(f"Diferença:                    +{np.mean(last_ages) - np.mean(first_ages):.1f} anos")
    
    first_bp = [r['payload']['trestbps'] for r in first_20]
    last_bp = [r['payload']['trestbps'] for r in last_20]
    print(f"\nPressão arterial (primeiras 20): {np.mean(first_bp):.1f} mmHg (drift baixo)")
    print(f"Pressão arterial (últimas 20):   {np.mean(last_bp):.1f} mmHg (drift alto)")
    print(f"Diferença:                       +{np.mean(last_bp) - np.mean(first_bp):.1f} mmHg")
    
    # Salva resultados para análise posterior
    output_file = "simulation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Resultados salvos em: {output_file}")
    print(f"✓ Logs de requisições disponíveis em: flask-app/requests.log")
    print("\nPróximo passo: Analisar data drift com visualizações customizadas")
    print("=" * 80)


if __name__ == "__main__":
    main()