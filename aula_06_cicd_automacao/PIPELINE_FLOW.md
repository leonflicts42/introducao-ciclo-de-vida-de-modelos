# Diagrama do Fluxo CI/CD - Aula 06

## 📊 Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESENVOLVEDOR                                 │
│  - Modifica código (train.py, preprocessing.py)                 │
│  - Ajusta hiperparâmetros                                        │
│  - Testa localmente                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ git commit & push
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GITHUB REPOSITORY                             │
│  - Detecta alterações na branch main                            │
│  - Trigger: paths em aula_06_cicd_automacao/** ou data/**       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Inicia GitHub Actions
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              JOB 1: TESTES (test)                                │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Setup Python 3.10                              │          │
│  │ 3. Instalar dependências (requirements.txt)       │          │
│  │ 4. Executar: python test_pipeline.py             │          │
│  │                                                    │          │
│  │ Testes incluem:                                   │          │
│  │  ✓ MissingValueImputer                           │          │
│  │  ✓ CategoricalEncoder                            │          │
│  │  ✓ FeatureEngineer                               │          │
│  │  ✓ Pipeline consistency                          │          │
│  │  ✓ Data loading & quality                        │          │
│  └───────────────────────────────────────────────────┘          │
│                         │                                        │
│                         │ Se falhar → STOP ❌                    │
│                         │ Se passar → Continua ✅                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         JOB 2: TREINAR BASELINE (train-baseline)                 │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Setup Python 3.10                              │          │
│  │ 3. Instalar dependências                          │          │
│  │ 4. Executar: python train.py --model-type baseline│          │
│  │                                                    │          │
│  │ Pipeline de Treinamento:                          │          │
│  │  ├─ Carregar dados (heart_disease_uci.csv)      │          │
│  │  ├─ Split train/test (80/20, stratify)          │          │
│  │  ├─ Criar pipeline sklearn:                      │          │
│  │  │   ├─ MissingValueImputer                     │          │
│  │  │   ├─ CategoricalEncoder                      │          │
│  │  │   ├─ FeatureEngineer                         │          │
│  │  │   ├─ StandardScaler                          │          │
│  │  │   └─ RandomForestClassifier(baseline params)│          │
│  │  ├─ Treinar modelo                               │          │
│  │  ├─ Avaliar métricas                             │          │
│  │  └─ Logar no MLflow:                             │          │
│  │       ├─ Parâmetros (n_estimators, max_depth...) │          │
│  │       ├─ Métricas (accuracy, precision, recall...)│          │
│  │       ├─ Modelo completo (pipeline)             │          │
│  │       └─ Se accuracy >= 0.75 → Model Registry   │          │
│  └───────────────────────────────────────────────────┘          │
│                         │                                        │
│                         │ Upload artifacts (mlruns/)             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│       JOB 3: TREINAR OTIMIZADO (train-optimized)                 │
│                  [CONDICIONAL]                                   │
│  Executa se:                                                     │
│   - workflow_dispatch (manual) OU                                │
│   - commit message contém "[train-optimized]"                    │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Setup Python 3.10                              │          │
│  │ 3. Instalar dependências                          │          │
│  │ 4. Executar: python train.py --model-type optimized│         │
│  │                                                    │          │
│  │ Diferenças vs Baseline:                           │          │
│  │  - n_estimators: 124 (vs 100)                    │          │
│  │  - max_depth: 15 (vs None)                       │          │
│  │  - max_features: 2 (vs 'sqrt')                   │          │
│  │  - min_samples_split: 10 (vs 2)                  │          │
│  │                                                    │          │
│  │ MLflow:                                           │          │
│  │  ├─ Log params, metrics, model                   │          │
│  │  ├─ Se accuracy >= 0.75 → Model Registry         │          │
│  │  └─ Alias "Production" → versão otimizada        │          │
│  └───────────────────────────────────────────────────┘          │
│                         │                                        │
│                         │ Upload artifacts (mlruns/)             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      JOB 4: COMPARAR E PROMOVER (compare-and-promote)            │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 1. Checkout código                                 │          │
│  │ 2. Exibir mensagem de sucesso                     │          │
│  │ 3. Instruções para visualizar no MLflow UI        │          │
│  └───────────────────────────────────────────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLFLOW TRACKING                               │
│  Experiment: heart-disease-cicd                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │ Run 1: baseline_random_forest                    │            │
│  │  - Parameters: n_estimators=100, max_depth=None │            │
│  │  - Metrics: test_accuracy=0.82, ...             │            │
│  │  - Artifacts: model/ (pipeline completo)        │            │
│  └─────────────────────────────────────────────────┘            │
│  ┌─────────────────────────────────────────────────┐            │
│  │ Run 2: optimized_random_forest                   │            │
│  │  - Parameters: n_estimators=124, max_depth=15   │            │
│  │  - Metrics: test_accuracy=0.85, ...             │            │
│  │  - Artifacts: model/ (pipeline completo)        │            │
│  └─────────────────────────────────────────────────┘            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MLFLOW MODEL REGISTRY                           │
│  Model: heart-disease-model                                      │
│  ┌─────────────────────────────────────────────────┐            │
│  │ Version 1: baseline                              │            │
│  │  - Source: Run 1                                 │            │
│  │  - Stage: None                                   │            │
│  │  - Accuracy: 0.82                                │            │
│  └─────────────────────────────────────────────────┘            │
│  ┌─────────────────────────────────────────────────┐            │
│  │ Version 2: optimized                    ⭐       │            │
│  │  - Source: Run 2                                 │            │
│  │  - Alias: Production                             │            │
│  │  - Accuracy: 0.85                                │            │
│  └─────────────────────────────────────────────────┘            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (Futuro)                           │
│  - Servir modelo via MLflow Model Serving                        │
│  - Deploy em container (Docker)                                  │
│  - API REST para inferência                                     │
│  - Monitoramento com Evidently (Aula 05)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo de Decisão

```
┌────────────────┐
│  Commit/Push   │
└───────┬────────┘
        │
        ▼
┌────────────────┐      ❌ Falhou
│  Testes        ├──────────────────► STOP (Notificar dev)
└───────┬────────┘
        │ ✅ Passou
        ▼
┌────────────────┐      Accuracy < 0.60
│  Train Baseline├──────────────────► STOP (Pipeline falha)
└───────┬────────┘
        │ Accuracy >= 0.75
        ├──────────────────────────► Registra no Model Registry
        │
        │ [train-optimized] no commit?
        ▼
┌─────────┴──────┐
│  Sim   │  Não  │
└────┬───┴───┬───┘
     │       │
     │       └──────────────────────► FIM
     ▼
┌────────────────┐      Accuracy < 0.60
│ Train Optimized├──────────────────► STOP (Pipeline falha)
└───────┬────────┘
        │ Accuracy >= 0.75
        ├──────────────────────────► Registra no Model Registry
        │
        │ Optimized > Baseline?
        ▼
┌─────────┴──────┐
│  Sim   │  Não  │
└────┬───┴───┬───┘
     │       │
     │       └──────────────────────► Mantém Baseline como Production
     ▼
Set Alias "Production" para Optimized
     │
     ▼
┌────────────────┐
│  FIM - Sucesso │
└────────────────┘
```

---

## 📂 Artefatos Gerados

```
aula_06_cicd_automacao/
├── mlruns/                           # Tracking local do MLflow
│   └── <experiment_id>/
│       ├── <run_id_baseline>/
│       │   ├── artifacts/
│       │   │   └── model/           # Pipeline completo (pkl)
│       │   ├── metrics/
│       │   │   ├── test_accuracy
│       │   │   ├── test_precision
│       │   │   └── ...
│       │   ├── params/
│       │   │   ├── n_estimators
│       │   │   ├── max_depth
│       │   │   └── ...
│       │   └── tags/
│       └── <run_id_optimized>/
│           └── ... (mesma estrutura)
└── preprocessing.py                  # Logado junto com o modelo
```

---

## 🎯 Critérios de Promoção

| Condição | Ação |
|----------|------|
| `test_accuracy < 0.60` | ❌ Pipeline falha, não loga no MLflow |
| `0.60 <= test_accuracy < 0.75` | ⚠️ Loga no MLflow, NÃO registra no Model Registry |
| `test_accuracy >= 0.75` | ✅ Loga no MLflow E registra no Model Registry |
| `optimized > baseline` | 🏆 Alias "Production" → modelo otimizado |

---

## 🔔 Notificações (Futuro)

O pipeline pode ser estendido para enviar notificações:
- ✉️ Email em caso de falha
- 💬 Slack quando novo modelo é promovido
- 📊 Dashboard com histórico de métricas
- 🚨 Alertas se acurácia cair abaixo de threshold

---

## 📚 Boas Práticas Implementadas

✅ **Separação de responsabilidades**: preprocessing.py, train.py, test_pipeline.py  
✅ **Testes automatizados**: Validação antes do treinamento  
✅ **Versionamento**: Git + MLflow Model Registry  
✅ **Reprodutibilidade**: random_state=42, code_paths no MLflow  
✅ **Documentação**: README, QUICKSTART, comentários no código  
✅ **CI/CD declarativo**: GitHub Actions YAML  
✅ **Model Registry**: Gestão de versões e aliases  

---

Este diagrama ilustra todo o fluxo do pipeline de CI/CD implementado na Aula 06! 🚀
