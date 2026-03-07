# Aula 7: Governança de Modelos e Ciclo de Feedback

## Objetivo
Explorar governança e fairness em um cenário didático de **aprovação de crédito**, documentando o modelo, monitorando vieses regionais e simulando feedback.

## Conceitos
- **Model Cards**: Documentação estruturada de modelos (framework Google)
- **Fairlearn**: Biblioteca Microsoft para avaliar e mitigar viés em modelos
- **Ciclo de Feedback**: Processo de coleta e incorporação de feedback de usuários

## Estrutura do Exercício
1. Gerar um dataset sintético de propostas de crédito (regiões Norte x Sul)
2. Treinar um classificador e diagnosticar viés regional
3. Aplicar mitigação com Fairlearn (ExponentiatedGradient + Demographic Parity)
4. Criar um Model Card completo
5. Simular um ciclo de feedback e registrar tudo no MLflow

## Como executar

```bash
# Ativar ambiente virtual
source ../venv/bin/activate

# Instalar dependência adicional (Fairlearn)
pip install fairlearn

# Abrir Jupyter Notebook
jupyter notebook exercicio_governanca_feedback.ipynb
```

## Métricas de Fairness avaliadas
- **Accuracy por grupo**: Desempenho separado por região
- **Selection Rate**: Taxa de aprovações por região
- **False Positive Rate**: Taxa de falsos positivos por região
- **False Negative Rate**: Taxa de falsos negativos por região

## Artifacts gerados
- `model_card.json`: Documentação estruturada do modelo
- `feedback_log.csv`: Histórico de feedbacks simulados
- MLflow run com métricas de governança

## Referências
- [Google Model Cards](https://modelcards.withgoogle.com/about)
- [Microsoft Fairlearn](https://fairlearn.org/)
- [Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
