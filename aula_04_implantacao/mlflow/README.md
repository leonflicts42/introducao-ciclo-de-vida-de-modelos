# Pipeline Completo de Machine Learning com MLflow

Este notebook demonstra a **melhor prática** para deployment de modelos: criar um **Pipeline sklearn completo** que inclui todas as etapas de pré-processamento como transformers customizados.

## 🎯 Objetivo

Demonstrar como integrar pré-processamento, feature engineering e modelo em um único pipeline sklearn, permitindo que dados **raw** sejam enviados diretamente para inferência.

## 📋 Componentes do Pipeline

### ⚠️ Pré-requisito: Limpeza de Dados
**Antes** de passar dados ao pipeline, sempre remova as colunas de metadados:
- `id`: Identificador único do paciente
- `dataset`: Origem dos dados
- `num`: Variável target raw (0-4)
- `target`: Variável alvo (apenas para treinamento)

```python
columns_to_drop = ['id', 'dataset', 'num', 'target']
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
```

### 1. **CategoricalEncoder**
Aplica encoding categórico e one-hot encoding:
- Mapeia valores numéricos para labels descritivos
- Aplica one-hot encoding com `drop_first=True`
- Colunas: `sex`, `cp`, `restecg`, `slope`, `thal`

### 2. **FeatureEngineer**
Cria features derivadas baseadas em conhecimento de domínio:
- `age_squared`: Relação quadrática com idade
- `age_decade`: Faixa etária por década
- `cholesterol_to_age`: Razão colesterol/idade
- `max_hr_pct`: Percentual da frequência cardíaca máxima teórica
- `bp_chol_ratio`: Razão pressão arterial/colesterol
- `fbs_flag`, `exang_flag`: Flags binárias
- `stress_index`: Índice de estresse cardiovascular
- `risk_interaction`: Interação idade × ST depression
- `high_st_depression_flag`: Flag para ST depression elevada

### 3. **StandardScaler**
Normalização de features numéricas

### 4. **RandomForestClassifier**
Modelo de classificação final

## 🔄 Fluxo de Dados

```
Dados RAW (CSV original)
        ↓
  [Remoção de colunas metadados: id, dataset, num, target]
        ↓
  ┌─────────── PIPELINE SKLEARN ───────────┐
  │                                         │
  │  CategoricalEncoder (encoding + OHE)    │
  │            ↓                            │
  │  FeatureEngineer (features derivadas)   │
  │            ↓                            │
  │  StandardScaler (normalização)          │
  │            ↓                            │
  │  RandomForestClassifier (predição)      │
  │                                         │
  └─────────────────────────────────────────┘
        ↓
  Resultado (0=Sem doença, 1=Com doença)
```

## ✅ Vantagens da Abordagem

### Consistência
O **mesmo código** de pré-processamento é usado em:
- Treinamento
- Validação
- Inferência local
- Inferência em produção (MLflow Serving)

### Simplicidade
```python
# Preparar dados (remover metadados)
raw_data = pd.read_csv('heart_disease_uci.csv')
X = raw_data.drop(columns=['id', 'dataset', 'num', 'target'], errors='ignore')

# Pipeline aplica todo o resto
prediction = pipeline.predict(X)
```

### Manutenibilidade
- Todas as transformações ficam **centralizadas** no pipeline
- Mudanças são feitas em um único lugar
- Fácil versionamento e rastreamento

### Deployment
- Pipeline completo é logado como uma **unidade atômica**
- MLflow gerencia todas as dependências
- Reprodutibilidade garantida

### Testabilidade
- Cada transformer pode ser testado isoladamente
- Pipeline completo pode ser testado end-to-end
- Fácil validação de comportamento

## 🚀 Como Usar

### 1. Treinar e Logar Pipeline
```python
# Preparar dados (remover metadados)
df = pd.read_csv('heart_disease_uci.csv')
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns=['id', 'dataset', 'num'])

X = df.drop(columns=['target'])
y = df['target']

# Criar pipeline (SEM ColumnDropper)
pipeline = Pipeline([
    ('categorical_encoding', CategoricalEncoder()),
    ('feature_engineering', FeatureEngineer()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Treinar
pipeline.fit(X, y)

# Logar no MLflow
mlflow.sklearn.log_model(pipeline, "model")
```

### 2. Carregar e Usar
```python
# Carregar do MLflow
loaded_model = mlflow.sklearn.load_model(model_uri)

# Preparar dados (remover metadados)
raw_data = pd.read_csv('heart_disease_uci.csv')
X = raw_data.drop(columns=['id', 'dataset', 'num', 'target'], errors='ignore')

# Predição
predictions = loaded_model.predict(X)
```

### 3. Servir via MLflow
```python
# Preparar dados (remover metadados antes de enviar)
raw_data = pd.read_csv('heart_disease_uci.csv')
X = raw_data.drop(columns=['id', 'dataset', 'num', 'target'], errors='ignore')

payload = {
    "dataframe_split": {
        "columns": X.columns.tolist(),
        "data": X.values.tolist()
    }
}

response = requests.post(endpoint_url, json=payload)
```

## 📊 Estrutura de Dados

### Entrada (Raw Data)
```
id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num
1, 63, Male, Cleveland, typical angina, 145, 233, TRUE, lv hypertrophy, 150, FALSE, 2.3, downsloping, 0, fixed defect, 0
```

### Saída
```
0 = Sem doença cardíaca
1 = Com doença cardíaca
```

## 🔧 Transformers Customizados

Todos os transformers seguem a interface sklearn:
- `fit(X, y=None)`: Aprende parâmetros (se necessário)
- `transform(X)`: Aplica transformação
- Herdam de `BaseEstimator` e `TransformerMixin`

## 📝 Notas Importantes

1. **Limpeza Pré-Pipeline**: Sempre remova colunas de metadados (`id`, `dataset`, `num`, `target`) **ANTES** de passar dados ao pipeline
2. **Ordem das Transformações**: A ordem dos steps no pipeline é crítica (encoding → feature engineering → scaling → modelo)
3. **Dados de Entrada**: Pipeline espera dados limpos (sem metadados) mas ainda no formato categórico original
4. **Signature**: MLflow signature é criada com dados limpos (sem metadados) na entrada

## 🎓 Conceitos Demonstrados

- ✅ Custom sklearn transformers
- ✅ Pipeline composition
- ✅ MLflow model logging
- ✅ Model registry
- ✅ Model serving
- ✅ Feature engineering como código
- ✅ Reprodutibilidade em ML

## 📚 Referências

- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Custom Transformers](https://scikit-learn.org/stable/developers/develop.html)
