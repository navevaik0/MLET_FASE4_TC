📈 LSTM Multi-Step Stock Forecast --- PETR4.SA
============================================

Este projeto implementa um pipeline completo para previsão de preços da ação **PETR4.SA** usando uma **Rede Neural LSTM** capaz de prever **5 passos à frente (t+1 a t+5)**.\
Todo o fluxo (coleta, preparação dos dados, modelagem, treinamento, avaliação e geração de previsões) é executado diretamente no **notebook principal (`Pipeline_Petr4.ipynb`)**.

Também acompanha uma **API FastAPI**, para servir previsões após o modelo estar treinado.

* * * * *

🚀 Funcionalidades
------------------

-   Coleta de dados da ação PETR4.SA (Yahoo Finance)

-   Pré-processamento dos dados

-   Criação das janelas de 60 timesteps

-   Normalização dos valores

-   Modelo LSTM multi-step (prevê 5 passos futuros)

-   Avaliação: MAE, RMSE, MAPE

-   Salvamento dos artefatos do modelo

-   Execução completa via **Jupyter Notebook**

-   API FastAPI para inferência:

    -   `/predict`

    -   `/predict/plot`

    -   `/last-window`

    -   `/health`

    -   `/compare-models`

* * * * *

📁 Estrutura do Projeto
-----------------------

``` bash
MLET_FASE4_TC/
│   requirements.txt
│
├── data/
│   ├── PETR4.SA.csv
│   └── processed_petr4_data.csv
│
├── models/
│   ├── lstm_multistep.h5
│   ├── lstm_multistep_test.keras
│   ├── lstm_multistep_tuned.keras
│   ├── scaler.save
│
├── api/
│   ├── main.py
│
├── kt_dir_test/
├── kt_test/
│
└── Pipeline_Petr4.ipynb
```

* * * * *

🔧 Instalação
-------------

### 1\. Criar ambiente virtual

**Windows (PowerShell)**

``` bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 2\. Instalar dependências

``` bash
pip install -r requirements.txt
```
* * * * *

▶️ Execução Principal (Notebook)
--------------------------------

Toda a execução do projeto ocorre no notebook:

`Pipeline_Petr4.ipynb`

No notebook você encontrará:

-   coleta dos dados

-   limpeza e preparação

-   criação das janelas

-   normalização

-   arquitetura LSTM

-   treinamento

-   avaliação

-   previsões (t+1 a t+5)

-   salvamento do modelo e scaler

Após isso, os artefatos ficam disponíveis na pasta `models/`.

* * * * *

▶️ Execução da API
---------------------------

Após treinar o modelo via notebook, você pode iniciar a API:

### 1\. Iniciar a API

``` bash
uvicorn api.main:app --host 0.0.0.0 --port 8000`
```

### 2\. Acessar a documentação (Swagger)

``` bash
http://localhost:8000/docs
```
* * * * *

🧪 Endpoints Disponíveis
------------------------

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | Verifica se a API está online |
| GET | `/last-window` | Exibe a última janela usada no modelo |
| POST | `/predict` | Retorna previsões t+1 a t+5 |
| POST | `/predict/plot` | Retorna gráfico Base64 |
| GET | `/compare-models` | Lista e compara os modelos disponíveis |

* * * * *

🤖 Arquitetura do Modelo LSTM
-----------------------------

-   Janela de entrada: **60 timesteps**

-   Previsão para: **5 passos futuros**

-   Duas camadas LSTM empilhadas

-   Camada Dense final para saída multi-step

-   Otimizador: **Adam**

-   Loss: **MSE**

-   Métricas: **MAE, RMSE, MAPE**

* * * * *

🧑‍💻 Autores
-------------

Projeto desenvolvido como parte do **Tech Challenge - Fase 4 (FIAP)**.

-   **Erick Navevaiko**

-   **Pedro Paolielo**
