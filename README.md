# 📈 LSTM Multi-Step Stock Forecast — PETR4.SA

Este projeto implementa um pipeline completo para previsão de preços da ação **PETR4.SA** usando uma **Rede Neural LSTM** capaz de prever **5 passos à frente (t+1 a t+5)**.  
Inclui coleta de dados, pré-processamento, treinamento, avaliação e uma **API FastAPI** para servir previsões em produção.

---

## 🚀 Funcionalidades

- Coleta automática dos dados (Yahoo Finance)
- Limpeza e normalização da série temporal
- Criação de janelas de 60 timesteps
- Modelo LSTM com previsão multi-step
- Avaliação (MAE, RMSE, MAPE)
- Servidor FastAPI para inferência
- Script para testes locais
- Suporte a Docker

---

## 📁 Estrutura do Projeto

```
FASE4_TC/
│   README.md
│   requirements.txt
│
├── data/
│   └── PETR4.SA.csv
│
├── models/
│   ├── lstm_multistep.h5
│   ├── lstm_multistep.keras
│   └── scaler.save
│
├── src/
│   ├── data_collection.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── api/
│       └── main.py
│
└── examples/
    └── run_predict.py
```

---

## 🔧 Instalação

### 1. Criar ambiente virtual

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

---

## 📥 Coletar dados PETR4.SA

```bash
python src/data_collection.py
```

Arquivo gerado em:
```
data/PETR4.SA.csv
```

---

## 🧠 Treinar o modelo LSTM

```bash
python -m src.train
```

O script irá:

- Ler o CSV  
- Criar sequências de 60 timesteps  
- Preparar horizonte de 5 passos  
- Treinar o modelo  
- Avaliar  
- Salvar arquivos em `models/`:

```
models/lstm_multistep.keras
models/lstm_multistep.h5
models/scaler.save
```

---

## 🔮 Testar previsão local

```bash
python examples/run_predict.py
```

Exemplo de saída:
```
Previsões (t+1 a t+5): [...]
```

---

## 🌐 Subir API FastAPI

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Acesse:

- http://localhost:8000  
- http://localhost:8000/docs

---

## 🎯 Exemplo de chamada ao endpoint `/predict`

**Payload mínimo (60 valores):**
```json
{
  "recent_closes": [10.1, 10.2, 10.3, ... 60 valores ...]
}
```

**Resposta:**
```json
{
  "predicted": [valor_t1, valor_t2, valor_t3, valor_t4, valor_t5]
}
```

---

## 🐳 Docker (Opcional)

### Build
```bash
docker build -t lstm-api .
```

### Run
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models lstm-api
```

---

## 🧪 Testes rápidos

```bash
curl http://localhost:8000/health
```

---

## ⚠️ Problemas comuns

### "Provide at least 60 closing prices"
Você enviou menos de 60 preços.

### Erro ao carregar `lstm_multistep.h5`
A API tenta automaticamente:
1. `models/lstm_multistep.keras`  
2. `models/lstm_multistep.h5`

### "ModuleNotFoundError: src"
Execute sempre da raiz:
```bash
python -m src.train
```

---

## 📝 Tecnologias

- Python 3.10+
- TensorFlow / Keras
- NumPy / Pandas
- FastAPI
- Yahoo Finance API (yfinance)
- Docker

---

## 🎓 Finalidade Acadêmica

Projeto desenvolvido para o **Tech Challenge – Fase 4 da FIAP**, demonstrando:

- Manipulação de séries temporais  
- Modelos LSTM multi-step  
- Deploy via API  
- Boas práticas de engenharia de Machine Learning  

---

## 📬 Autores

**Erick Navevaiko e Pedro Paolielo**  
FIAP – Pós-Tech  
Tech Challenge Fase 4
