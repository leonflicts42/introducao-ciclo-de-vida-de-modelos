# Aula 4: Implantação de Modelos (Deployment)

## Descrição
Implante modelos em produção usando diferentes estratégias.

## Objetivos
- Registrar modelos no MLFlow Model Registry
- Versionar modelos adequadamente
- Servir modelos via API REST
- Implementar estratégias de deployment

## Exercício
Execute o notebook `exercicio_deployment.ipynb`

## API Flask de Predição (Opcional)
Além do notebook, você pode subir uma API simples em Flask para servir o modelo treinado na Aula 3.

### Pré-requisitos
- Ambiente Python do curso ativado
- Dependências instaladas: `pip install -r requirements.txt` (agora inclui Flask)
- Modelo salvo em `models/best_random_forest.joblib` (gerado na Aula 3)

### Executando localmente
No diretório raiz do repositório:

```bash
export FLASK_APP=aula_04_implantacao/flask-app/app.py
python aula_04_implantacao/flask-app/app.py
```

Endpoints:
- `GET /` — Health check
- `POST /heart-disease-predict` — Recebe JSON e retorna predição

Formato de entrada:
- Você pode enviar já no formato pré-processado (colunas one-hot como `cp_atypical angina`, `restecg_normal`, etc.)
- OU enviar colunas brutas `cp`, `restecg`, `slope`, `thal`, `sex` que serão convertidas automaticamente.

Mapeamentos aceitos (valores inteiros):
- `cp`: valores (0..3 ou 1..4) mapeados para: típica angina, atípica angina, não-anginal, assintomático (assintomático é agrupado em não-anginal porque a coluna específica não existe no modelo)
- `restecg`: 0 → normal, 1 → ST-T abnormality, 2 → LVH (agrupado em ST-T abnormality porque a coluna LVH não existe no modelo)
- `slope`: 0/1 → upsloping, 1/2 → flat, 2/3 → downsloping (agrupado em flat para manter o número de colunas)
- `thal`: 0/1 → normal, 1/2 → fixed defect (agrupado em normal porque a coluna fixed defect não existe), 2/3 → reversable defect
- `sex`: 0 → Female, 1 → Male

Se uma coluna já estiver one-hot, ela é mantida. Colunas esperadas e ausentes são preenchidas com 0/False.

Ordem das features: a API alinha automaticamente as colunas ao mesmo conjunto e ordem usadas no treinamento do modelo. Quaisquer colunas extras são descartadas e faltantes são adicionadas com 0.

### Exemplo de requisição
```bash
curl -X POST http://localhost:5000/heart-disease-predict \
	-H "Content-Type: application/json" \
	-d '{
				"age": 57,
				"sex": 1,
				"cp": 0,
				"trestbps": 130,
				"chol": 236,
				"fbs": 0,
				"restecg": 1,
				"thalch": 174,
				"exang": 0,
				"oldpeak": 0.0,
				"slope": 1,
				"ca": 1,
				"thal": 2
			}'
```

Também suporta batch via `instances`:

```json
{
	"instances": [
		{ "age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 236, "fbs": 0, "restecg": 1, "thalch": 174, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 1, "thal": 2 },
		{ "age": 62, "sex": 0, "cp": 2, "trestbps": 120, "chol": 281, "fbs": 0, "restecg": 1, "thalch": 103, "exang": 1, "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 3 }
	]
}
```

Observação: a API replica o feature engineering essencial da Aula 3 (ex.: `age_squared`, `bp_chol_ratio`, etc.) quando as colunas base são informadas. Caso você já envie essas colunas engenheiradas, elas serão respeitadas.
