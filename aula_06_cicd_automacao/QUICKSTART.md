# 🚀 Quick Start: CI/CD Pipeline para Machine Learning

Este guia rápido mostra como usar o pipeline de CI/CD implementado na Aula 06.

## 📋 Pré-requisitos

- Python 3.8+
- Dependências instaladas: `pip install -r requirements.txt`
- Git configurado
- (Opcional) Conta no GitHub para usar GitHub Actions

---

## 🏃 Execução Local (Para Testar)

### 1. Rodar Testes
```bash
cd aula_06_cicd_automacao
python test_pipeline.py
```

### 2. Treinar Modelo Baseline
```bash
python train.py --model-type baseline --min-accuracy 0.75
```

### 3. Treinar Modelo Otimizado
```bash
python train.py --model-type optimized --min-accuracy 0.75
```

### 4. Visualizar no MLflow
```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

## 🔄 Pipeline Automático (GitHub Actions)

### Cenário 1: Treinar Apenas Baseline
```bash
git add aula_06_cicd_automacao/train.py
git commit -m "Ajuste no código de treinamento"
git push origin main
```
→ Pipeline executa: **testes + baseline**

### Cenário 2: Treinar Baseline + Otimizado
```bash
git add aula_06_cicd_automacao/train.py
git commit -m "[train-optimized] Nova versão com ajustes"
git push origin main
```
→ Pipeline executa: **testes + baseline + otimizado**

### Cenário 3: Execução Manual
1. Acesse GitHub → **Actions** → **CI/CD Pipeline - Heart Disease Model**
2. Clique em **Run workflow**
3. Selecione o tipo de modelo
4. Clique em **Run workflow**

---

## 📊 Monitorar Resultados

### No GitHub:
- Vá para **Actions** no repositório
- Clique no workflow executado
- Veja logs de cada job (test, train-baseline, train-optimized)

### No MLflow:
- Execute `mlflow ui` localmente
- Ou acesse servidor remoto (se configurado)
- Compare métricas entre runs
- Veja versões no Model Registry

---

## 🎓 Para Alunos: Exercício Prático

### Passo 1: Executar Localmente
```bash
cd aula_06_cicd_automacao
python test_pipeline.py
python train.py --model-type baseline
mlflow ui
```

### Passo 2: Modificar Hiperparâmetros
Edite `train.py` e altere `BASELINE_PARAMS`:
```python
BASELINE_PARAMS = {
    'n_estimators': 150,  # Era 100
    'max_depth': 10,      # Era None
    'random_state': 42
}
```

### Passo 3: Testar Nova Versão
```bash
python train.py --model-type baseline
# Compare as métricas no MLflow UI
```

### Passo 4: Fazer Commit e Push
```bash
git add aula_06_cicd_automacao/train.py
git commit -m "Ajuste de hiperparâmetros baseline"
git push origin main
```

### Passo 5: Acompanhar Pipeline
- Vá para GitHub Actions
- Veja o pipeline executando automaticamente
- Confira os resultados

---

## 🔧 Troubleshooting

### Problema: Testes falhando
```bash
# Rode os testes com verbose
python -m pytest test_pipeline.py -v
```

### Problema: MLflow não encontrado
```bash
# Instale novamente
pip install mlflow
```

### Problema: Dataset não encontrado
```bash
# Verifique o caminho
ls ../data/heart_disease_uci.csv
```

### Problema: GitHub Actions não executando
- Verifique que o arquivo `.github/workflows/cicd-pipeline.yml` existe
- Confirme que está fazendo push na branch `main`
- Verifique que os paths no workflow correspondem aos arquivos alterados

---

## 📚 Recursos Adicionais

- **README.md**: Documentação completa da Aula 06
- **test.ipynb**: Notebook demonstrativo interativo
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **GitHub Actions Docs**: https://docs.github.com/en/actions

---

## ✅ Checklist de Aprendizagem

- [ ] Executei os testes localmente
- [ ] Treinei o modelo baseline
- [ ] Treinei o modelo otimizado
- [ ] Visualizei resultados no MLflow UI
- [ ] Comparei métricas entre modelos
- [ ] Modifiquei hiperparâmetros e re-treinei
- [ ] Fiz commit e push para trigger do pipeline
- [ ] Acompanhei execução no GitHub Actions
- [ ] Verifiquei logs do pipeline
- [ ] Entendi como funciona o Model Registry

---

**Dúvidas?** Consulte o README.md completo ou o notebook test.ipynb!
