# Tech Challenge – Fase 4  
## Deep Learning para Previsão de Séries Temporais Financeiras

**Pós Tech | Machine Learning Engineering – FIAP**

---

## 1. Introdução

Este projeto foi desenvolvido como parte do **Tech Challenge da Fase 4** da Pós Tech em **Machine Learning Engineering (FIAP)**.  
O desafio consiste na construção de um **modelo de Deep Learning baseado em LSTM (Long Short-Term Memory)** para **previsão do preço de fechamento de uma ação da bolsa de valores**, abrangendo **toda a pipeline de Machine Learning**, desde a coleta dos dados até o **deploy do modelo em uma API REST**.

O trabalho demonstra a aplicação prática de redes neurais recorrentes para séries temporais financeiras, bem como boas práticas de engenharia de Machine Learning voltadas para ambientes de produção.

---

## 2. Objetivo

Os principais objetivos deste projeto são:

- Desenvolver um modelo preditivo utilizando **LSTM** para séries temporais financeiras  
- Coletar e preparar dados históricos de preços de ações  
- Treinar, validar e avaliar o desempenho do modelo  
- Salvar o modelo treinado para reutilização  
- Disponibilizar o modelo por meio de uma **API RESTful**  
- Apresentar conceitos de escalabilidade e monitoramento em produção  

---

## 3. Tecnologias Utilizadas

- **Python 3**
- **TensorFlow / Keras**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **yFinance**
- **FastAPI**
- **Uvicorn**
- **Jupyter Notebook**

---

## 4. Coleta e Pré-processamento dos Dados

Os dados utilizados neste projeto correspondem a **preços históricos de ações**, obtidos a partir do **Yahoo Finance** por meio da biblioteca `yfinance`.

### Etapas realizadas:

1. Coleta dos dados históricos (Open, High, Low, Close, Volume)  
2. Seleção do **preço de fechamento (Close)** como variável alvo  
3. Tratamento de valores ausentes  
4. Normalização dos dados utilizando **MinMaxScaler**  
5. Criação de janelas temporais (sliding window) para adaptação ao modelo LSTM  
6. Separação dos dados em conjuntos de treino e teste respeitando a ordem temporal  

---

## 5. Desenvolvimento do Modelo LSTM

O modelo foi desenvolvido utilizando **Redes Neurais Recorrentes do tipo LSTM**, que são especialmente adequadas para capturar dependências de longo prazo em séries temporais.

### Características do modelo:

- Janelas de entrada com **60 timesteps**
- Arquitetura com camadas LSTM empilhadas
- Camada densa final para saída de regressão
- Função de perda adequada para problemas de regressão
- Otimizador **Adam**

O treinamento foi realizado exclusivamente com dados históricos passados, evitando vazamento de informação temporal.

---

## 6. Avaliação do Modelo

O desempenho do modelo foi avaliado utilizando métricas amplamente adotadas em problemas de regressão para séries temporais financeiras:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **MAPE (Mean Absolute Percentage Error)**

Essas métricas permitem avaliar a precisão das previsões e o erro médio em relação aos valores reais do preço de fechamento da ação.

---

## 7. Salvamento e Versionamento do Modelo

Após o treinamento, o modelo foi salvo em formato reutilizável (`.h5` / `.keras`), possibilitando:

- Carregamento rápido para inferência
- Reutilização do modelo sem necessidade de novo treinamento
- Integração direta com a API de deploy

Além do modelo, o **scaler utilizado na normalização** também foi salvo para garantir consistência nas previsões futuras.

---

## 8. Deploy do Modelo – API REST

O modelo treinado foi disponibilizado por meio de uma **API REST**, desenvolvida com **FastAPI**, permitindo que usuários realizem previsões a partir de dados históricos fornecidos via requisição HTTP.

### Funcionalidades da API:

- Carregamento do modelo treinado
- Recebimento de dados históricos para inferência
- Retorno das previsões de preços futuros
- Documentação automática via Swagger

---

## 9. Escalabilidade e Monitoramento (Conceitual)

Em um ambiente de produção real, a solução poderia ser monitorada utilizando:

- Logs estruturados para rastreamento de requisições  
- Monitoramento de tempo de resposta da API  
- Monitoramento de consumo de CPU e memória  
- Ferramentas como **Prometheus**, **Grafana** ou serviços de monitoramento em nuvem  

Essas práticas garantiriam maior confiabilidade, escalabilidade e observabilidade do modelo em produção.

---

## 10. Estrutura do Projeto

```bash
MLET_FASE4_TC/
│   requirements.txt
│
├── data/
│   └── arquivos de dados históricos e processados
│
├── models/
│   └── modelos treinados e artefatos auxiliares
│
├── api/
│   └── código da API REST
│
├── notebooks/
│   └── notebooks de desenvolvimento e experimentação
│
└── README.md
```

## 11. Como Executar o Projeto

Esta seção descreve o passo a passo necessário para executar o projeto localmente, desde a clonagem do repositório até a inicialização da API.

### 11.1 Clonar o Repositório

```bash
git clone https://github.com/navevaik0/MLET_FASE4_TC.git
cd MLET_FASE4_TC
```

### 11.2 Criar Ambiente Virtual (Opcional, porém Recomendado)
```
python -m venv venv
```
Ative o ambiente virtual conforme o sistema operacional.

Windows
```
venv\Scripts\activate
```
Linux / macOS
```
source venv/bin/activate
```
###11.3 Instalar as Dependências
```
pip install -r requirements.txt
```
## 12. Execução do Pipeline de Machine Learning

O pipeline completo de Machine Learning é executado por meio do Jupyter Notebook principal do projeto.
Durante a execução do notebook são realizadas as seguintes etapas:
- Coleta dos dados históricos da ação
- Limpeza e pré-processamento dos dados
- Criação das janelas temporais para séries temporais
- Normalização dos dados
- Construção da arquitetura do modelo LSTM
- Treinamento do modelo
- Avaliação do desempenho utilizando métricas apropriadas
- Geração de previsões
- Salvamento do modelo treinado e dos artefatos auxiliares
Após a execução do notebook, os modelos treinados e os objetos de pré-processamento ficam disponíveis para utilização pela API.

## 13. Execução da API de Deploy

Após a conclusão do treinamento do modelo, a API REST pode ser inicializada para realização de inferências.

### 13.1 Inicializar a API
```
uvicorn api.main:app --reload
```
A aplicação ficará disponível no endereço:
```
http://localhost:8000
```

### 13.2 Documentação Interativa da API

A documentação interativa da API, gerada automaticamente pelo FastAPI (Swagger), pode ser acessada em:
```
http://localhost:8000/docs
```
## 14. Considerações sobre Produção, Escalabilidade e Monitoramento

Em um ambiente de produção real, a solução pode ser aprimorada com as seguintes práticas:
- Implementação de logs estruturados para rastreamento de requisições
- Monitoramento do tempo de resposta da API
- Monitoramento do consumo de CPU e memória
- Versionamento de modelos para análise de desempenho ao longo do tempo
- Utilização de ferramentas de observabilidade como Prometheus, Grafana ou serviços de monitoramento em nuvem
Essas estratégias contribuem para maior confiabilidade, escalabilidade e robustez do sistema em produção.

## 15. Autores

- Erick Navevaiko
- Pedro Paolielo
