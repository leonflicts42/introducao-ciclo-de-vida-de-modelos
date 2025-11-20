# Monitoramento de Requisições do Modelo

Esta aula demonstra como implementar um processo simples de monitoramento/log das requisições ao endpoint de predição.
Cada requisição gera uma linha (JSON) em `requests.log` com os campos abaixo:

| Campo | Descrição |
|-------|-----------|
| `timestamp` | Momento em UTC em que a requisição foi processada (ISO8601). |
| `request_id` | UUID único para correlacionar logs e rastrear a requisição. |
| `request_payload` | Conteúdo bruto enviado pelo cliente (body original). |
| `response_payload` | Objeto de resposta enviado ao cliente (predições, probabilidades ou erro). |
| `status_code` | Código HTTP retornado. |
| `latency_ms` | Latência total da execução (recebimento até resposta) em milissegundos. |

## Uso

1. Ative o ambiente e instale dependências se ainda não fez:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
2. Execute a aplicação (na pasta `aula_05_monitoramento_manutencao/flask-app/`):
```bash
python app.py
```
3. Faça uma requisição de teste:
```bash
curl -X POST http://localhost:5000/heart-disease-predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalch":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```
4. Verifique o arquivo `requests.log` para visualizar as linhas de monitoramento.

## Variáveis de Ambiente
- `MODEL_PATH`: caminho alternativo para o modelo (`.joblib`).
- `REQUEST_LOG_PATH`: caminho alternativo para o arquivo de log (default = `requests.log` local).

## Extensões Possíveis
- Enviar estes logs para um sistema de mensageria (Kafka, RabbitMQ) ou observabilidade (ELK, Loki, etc.).
- Incorporar métricas agregadas (ex.: média de latência) via Prometheus.
- Enriquecer com usuário, IP, headers, etc.

Mantenha o foco em registrar eventos de requisição para posterior análise de performance e qualidade do modelo.
