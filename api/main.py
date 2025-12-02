"""
main.py - FastAPI para previsão multistep (PETR4.SA)
Endpoints:
 - /health
 - /last-window
 - /predict
 - /predict/plot
 - /compare-models

"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# tensorflow/keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configurações (ajuste se necessário)
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PROCESSED_CSV = DATA_DIR / "PETR4.SA.processed.csv"  # se existir, será usado
DEFAULT_SEQ_LEN = 60
DEFAULT_HORIZON = 5

app = FastAPI(
    title="Stock Forecast API (PETR4.SA)",
    description="API para previsão multistep (t+1..t+H) usando modelo LSTM",
    version="1.0.0",
)


# ---------------------------
# Pydantic models (entrada/saída)
# ---------------------------
class PredictRequest(BaseModel):
    """Entrada para /predict: permite seq_len e horizon opcionais e escolha do modelo."""
    seq_len: Optional[int] = Field(DEFAULT_SEQ_LEN, description="Comprimento da janela (n timesteps de entrada)")
    horizon: Optional[int] = Field(DEFAULT_HORIZON, description="Horizonte de previsão (n passos à frente)")
    model: Optional[str] = Field("auto", description="Modelo a usar: 'auto' | 'base' | 'tuned' | caminho personalizado")
    # opcionalmente poderia aceitar 'ticker' no futuro
    # ticker: Optional[str] = Field('PETR4.SA')


class PredictResponse(BaseModel):
    """Saída do /predict"""
    horizon: List[str]
    forecast: List[float]
    last_window: List[float]
    meta: Dict[str, Any] = {}


# ---------------------------
# Helpers: carregar modelo, scaler e dados
# ---------------------------
def find_model_path(prefer: Optional[str] = None) -> Optional[Path]:
    """
    Localiza um arquivo de modelo em models/.
    prefer: 'base' procura 'lstm_multistep.keras' ou 'lstm_multistep.h5'
            'tuned' procura 'lstm_multistep_tuned.keras' ou similar
            'auto' tenta tuned -> base -> qualquer .keras/.h5 no dir
            se for caminho absoluto/relativo existente, retorna esse
    """
    # se o usuário passou um caminho
    if prefer and prefer != "auto":
        p = Path(prefer)
        if p.exists():
            return p
        # se prefer for 'base' ou 'tuned', tratamos abaixo

    # padrões que usamos no projeto
    candidates = []
    if prefer == "base":
        candidates += ["models/lstm_multistep.keras", "models/lstm_multistep.h5", "models/lstm_multistep_test.keras"]
    elif prefer == "tuned":
        candidates += ["models/lstm_multistep_tuned.keras", "models/lstm_multistep_tuned.h5"]
    else:  # auto
        candidates += ["models/lstm_multistep_tuned.keras", "models/lstm_multistep_tuned.h5",
                       "models/lstm_multistep.keras", "models/lstm_multistep.h5",
                       "models/lstm_multistep_test.keras"]

    # procurar candidatos no disco
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p

    # se nada encontrado, procurar qualquer .keras ou .h5 em models dir
    if MODELS_DIR.exists():
        for ext in ("*.keras", "*.h5"):
            found = list(MODELS_DIR.glob(ext))
            if found:
                return found[0]
    return None


def load_model_and_scaler(model_path: Optional[Path] = None):
    """
    Carrega modelo Keras e scaler salvo (joblib) se existir.
    Retorna (model, scaler) ou (None, None) se não encontrar.
    """
    if model_path is None:
        model_path = find_model_path("auto")
    if model_path is None:
        return None, None

    try:
        # carregar modelo sem compilar para evitar problemas com métricas custom
        model = load_model(str(model_path), compile=False)
    except Exception as e:
        # tentativa alternativa: tensorflow pode suportar .keras/.h5; fornecer mensagem clara
        raise RuntimeError(f"Falha ao carregar o modelo {model_path}: {e}")

    # scaler
    scaler_path = MODELS_DIR / "scaler.save"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    return model, scaler


def load_processed_dataframe() -> pd.DataFrame:
    """
    Tenta carregar o CSV processado (com Close numérico). Lança HTTPException se não existir.
    """
    if not PROCESSED_CSV.exists():
        raise HTTPException(status_code=404, detail=f"CSV processado não encontrado: {PROCESSED_CSV}. Execute pré-processamento.")
    df = pd.read_csv(PROCESSED_CSV, index_col=0, parse_dates=True)
    # garantir Close numérico
    if "Close" not in df.columns:
        raise HTTPException(status_code=400, detail="Coluna 'Close' não encontrada no CSV processado.")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def create_multistep_input_from_series(series: np.ndarray, seq_len: int):
    """
    Recebe series (n,1) ou (n,) e retorna a última janela shape (1, seq_len, 1)
    """
    arr = np.asarray(series).reshape(-1, 1)
    if len(arr) < seq_len:
        raise ValueError(f"Série muito curta ({len(arr)}) para seq_len={seq_len}")
    last_window = arr[-seq_len:, 0]
    return last_window.reshape(1, seq_len, 1), last_window


def inverse_transform_preds(preds: np.ndarray, scaler):
    """
    Inverse transform previsões quando o scaler existe.
    preds: array shape (1,h) ou (n,h)
    """
    if scaler is None:
        # não temos scaler -> retornamos valores do modelo diretamente
        return preds.tolist()
    # scaler expects 2D array; transform each horizon column flattened
    orig_shape = preds.shape
    flat = preds.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    inv = inv.reshape(orig_shape)
    return inv.tolist()


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health", tags=["infra"], summary="Verificação de saúde da API", description="Retorna status básico para health-checks.")
def health():
    return {"status": "ok"}


@app.get("/last-window", summary="Última janela de Close", description="Retorna os últimos N valores de Close (padrão seq_len=60) do CSV processado.")
def last_window(seq_len: int = DEFAULT_SEQ_LEN):
    df = load_processed_dataframe()
    close = df["Close"].dropna().values
    if len(close) < seq_len:
        raise HTTPException(status_code=400, detail=f"Série muito curta ({len(close)}) para seq_len={seq_len}")
    last = close[-seq_len:].tolist()
    return {"seq_len": seq_len, "last_window": last, "rows": len(close)}


@app.post("/predict", response_model=PredictResponse, summary="Fazer previsão multistep",
          description="Gera previsões t+1..t+H usando o modelo salvo. Requer CSV pre-processado presente em data/PETR4.SA.processed.csv")
def predict(req: PredictRequest):
    # carregar modelo e scaler
    model_path = None
    if req.model and req.model not in ("auto", ""):
        # permitir nomes 'base'/'tuned' ou caminho
        model_path = find_model_path(req.model if req.model in ("base", "tuned") else None)
        # se req.model for caminho
        if req.model not in ("base", "tuned", "auto"):
            if Path(req.model).exists():
                model_path = Path(req.model)
    # fallback auto
    model, scaler = load_model_and_scaler(model_path)

    if model is None:
        raise HTTPException(status_code=404, detail="Modelo não encontrado. Treine e salve o modelo em models/ antes de usar /predict.")

    # carregar dados processados e extrair última janela
    df = load_processed_dataframe()
    close = df["Close"].dropna().values

    try:
        inp, last_window = create_multistep_input_from_series(close, req.seq_len)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # aplicar scaler se existir
    if scaler is not None:
        # scaler foi ajustado sobre shape (n,1) -> precisamos transformar a janela inteira
        inp_flat = last_window.reshape(-1, 1)
        inp_scaled = scaler.transform(inp_flat).reshape(1, req.seq_len, 1)
    else:
        inp_scaled = inp

    # prever
    preds = model.predict(inp_scaled)
    # modelos many-to-many tipicamente retornam shape (1, horizon)
    preds_list = inverse_transform_preds(preds, scaler)

    # montar nome de horizontes t+1..t+H
    horizon_names = [f"t+{i+1}" for i in range(preds.shape[1])]

    resp = {
        "horizon": horizon_names,
        "forecast": [float(x) for x in np.array(preds_list).reshape(-1).tolist()],
        "last_window": last_window.tolist(),
        "meta": {
            "model_path": str(model_path) if model_path else "auto",
            "seq_len": req.seq_len,
            "horizon": preds.shape[1]
        }
    }
    return JSONResponse(content=resp)


@app.get("/predict/plot", summary="Plot com histórico + previsões", description="Gera um PNG com o histórico de Close e a previsão t+1..t+H.")
def predict_plot(seq_len: int = DEFAULT_SEQ_LEN,
                 horizon: int = DEFAULT_HORIZON,
                 model: str = Query("auto", description="Modelo: auto|base|tuned|caminho")):
    # usar a rota /predict internamente (evitar duplicar lógica)
    req = PredictRequest(seq_len=seq_len, horizon=horizon, model=model)
    # chamar função predict (não via HTTP) para obter previsões
    try:
        predict_resp = predict(req)
        # predict_resp é JSONResponse -> extrair conteúdo
        content = predict_resp.body
        # FastAPI JSONResponse stores body as bytes; decodificar
        try:
            json_obj = json.loads(content.decode("utf-8"))
        except Exception:
            # se já for dict
            json_obj = predict_resp.body
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsão: {e}")

    # carregar df e preparar plot
    df = load_processed_dataframe()
    close = df["Close"].dropna()
    last_window = json_obj["last_window"]
    preds = json_obj["forecast"]
    horizon_names = json_obj["horizon"]

    # preparar eixo de datas: vamos usar índice do df e criar datas futuras fictícias baseadas em freq se possível
    dates = close.index
    last_date = dates[-1] if len(dates) > 0 else None
    # construir future dates: tentar inferir freq
    future_idx = []
    try:
        freq = pd.infer_freq(dates)
        if freq:
            future_idx = pd.date_range(start=last_date, periods=len(preds)+1, freq=freq)[1:]
        else:
            # se não conseguimos inferir, usar integer steps (positional)
            future_idx = list(range(len(close), len(close) + len(preds)))
    except Exception:
        future_idx = list(range(len(close), len(close) + len(preds)))

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(close.index, close.values, label="historical Close")
    # plot last window as thicker
    window_idx = pd.RangeIndex(start=len(close)-len(last_window), stop=len(close))
    try:
        # convert window_idx to corresponding dates
        plt.plot(close.index[-len(last_window):], last_window, linewidth=3, label="last window")
    except Exception:
        plt.plot(range(len(close)-len(last_window), len(close)), last_window, linewidth=3, label="last window")

    # plot predictions
    try:
        plt.plot(future_idx, preds, marker="o", linestyle="--", label="forecast")
    except Exception:
        plt.plot(range(len(close), len(close) + len(preds)), preds, marker="o", linestyle="--", label="forecast")

    plt.title("Histórico Close + Previsão")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/compare-models", summary="Comparar modelos base vs tuned",
         description="Retorna métricas MAE/RMSE por horizonte para modelos base e tunado. \
         Procura por arquivos models/metrics_base.json e models/metrics_tuned.json. \
         Se não encontrados, tenta calcular usando data/PETR4.SA.processed.csv e os modelos salvos.")
def compare_models():
    # 1) se existirem arquivos de métricas salvos, retornar direto
    metrics_base_path = MODELS_DIR / "metrics_base.json"
    metrics_tuned_path = MODELS_DIR / "metrics_tuned.json"

    if metrics_base_path.exists() and metrics_tuned_path.exists():
        try:
            base = json.loads(metrics_base_path.read_text(encoding="utf-8"))
            tuned = json.loads(metrics_tuned_path.read_text(encoding="utf-8"))
            return {"base": base, "tuned": tuned, "source": "files"}
        except Exception as e:
            # fallback para recálculo
            pass

    # 2) tentar recalc a partir dos modelos e do CSV processado
    # requisitos: processed csv + modelos base/tuned
    try:
        df = load_processed_dataframe()
    except HTTPException as e:
        raise HTTPException(status_code=404, detail="CSV processado não encontrado para cálculo de métricas. Execute pré-processamento.") from e

    # preparar sequências (usar seq_len e horizon padrão)
    seq_len = DEFAULT_SEQ_LEN
    horizon = DEFAULT_HORIZON

    series = df["Close"].dropna().values.reshape(-1, 1)
    # carregar scaler (se existir) e modelos
    # tentamos carregar base e tuned (se existirem)
    base_model_path = find_model_path("base")
    tuned_model_path = find_model_path("tuned")

    results = {}
    for label, mpath in (("base", base_model_path), ("tuned", tuned_model_path)):
        if mpath is None:
            results[label] = {"error": "modelo não encontrado"}
            continue
        try:
            model = load_model(str(mpath), compile=False)
        except Exception as e:
            results[label] = {"error": f"falha ao carregar modelo: {e}"}
            continue

        # carregar scaler (mesmo para ambos; assumimos que scaler comum foi salvo)
        scaler_path = MODELS_DIR / "scaler.save"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # criar sequências (mesma lógica do pipeline)
        # escala toda a série com scaler, se disponível
        if scaler is not None:
            scaled = scaler.transform(series)
        else:
            scaled = series

        X, y = [], []
        max_start = len(scaled) - seq_len - horizon + 1
        if max_start <= 0:
            results[label] = {"error": "série muito curta para seq_len/horizon"}
            continue
        for i in range(max_start):
            X.append(scaled[i:i + seq_len, 0])
            y.append(scaled[i + seq_len:i + seq_len + horizon, 0])
        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y)
        # split temporal: 70/15/15 (mesma lógica)
        n = len(X)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.15)
        X_test = X[val_end:]
        y_test = y[val_end:]
        if X_test.shape[0] == 0:
            results[label] = {"error": "sem dados de teste após split"}
            continue

        preds = model.predict(X_test)
        # inverse transform preds and y_test if scaler exists
        if scaler is not None:
            preds_un = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            y_un = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        else:
            preds_un = preds
            y_un = y_test

        # calcular MAE e RMSE por horizon
        maes, rmses = [], []
        for h in range(preds_un.shape[1]):
            mae = mean_absolute_error(y_un[:, h], preds_un[:, h])
            mse = mean_squared_error(y_un[:, h], preds_un[:, h])
            rms = float(np.sqrt(mse))
            maes.append(float(mae))
            rmses.append(rms)
        results[label] = {"mae": maes, "rmse": rmses}

    # salvar em arquivos para próxima vez (opcional)
    try:
        if "base" in results and isinstance(results["base"], dict) and "mae" in results["base"]:
            (MODELS_DIR / "metrics_base.json").write_text(json.dumps(results["base"], indent=2), encoding="utf-8")
        if "tuned" in results and isinstance(results["tuned"], dict) and "mae" in results["tuned"]:
            (MODELS_DIR / "metrics_tuned.json").write_text(json.dumps(results["tuned"], indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"base": results.get("base"), "tuned": results.get("tuned"), "source": "computed"}


# ---------------------------
# Mensagem de execução (opcional)
# ---------------------------
@app.on_event("startup")
def startup_event():
    # apenas log simples no startup
    print("API iniciada. Procurando por modelos em:", MODELS_DIR.resolve())
    m = find_model_path("auto")
    if m:
        print("Modelo encontrado:", m)
    else:
        print("Nenhum modelo encontrado automaticamente em models/.")


# Se for executado diretamente (dev), permite executar com: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)