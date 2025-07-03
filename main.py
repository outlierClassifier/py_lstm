import os
import re
import time
import datetime
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F

LOGGER_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGER_FMT)
logger = logging.getLogger("disruption-classifier")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR = "artifacts"
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW = 2_048       # Nº muestras por ventana
STRIDE = 512         # Paso de la ventana (75 % solape)

PATTERN = r"DES_(\d+)_(\d+)"

class Signal(BaseModel):
    fileName: str
    values: List[float]
    times: Optional[List[float]] = None
    length: Optional[int] = None

class Discharge(BaseModel):
    id: str
    times: Optional[List[float]] = None
    length: Optional[int] = None
    anomalyTime: Optional[float] = None
    signals: List[Signal]

class PredictionRequest(BaseModel):
    discharges: List[Discharge]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    executionTimeMs: float
    model: str
    details: Optional[Dict[str, Any]] = None

class TrainingOptions(BaseModel):
    epochs: Optional[int] = 10
    batchSize: Optional[int] = 128
    modelType: Optional[str] = "ensemble"  # cnn | lstm | transformer | ensemble
    learningRate: Optional[float] = 1e-3

class TrainingRequest(BaseModel):
    discharges: List[Discharge]
    options: Optional[TrainingOptions] = None

class TrainingMetrics(BaseModel):
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1Score: Optional[float] = None

class TrainingResponse(BaseModel):
    status: str
    message: Optional[str] = None
    trainingId: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    executionTimeMs: float

class MemoryInfo(BaseModel):
    total: float
    used: float

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    uptime: float
    memory: MemoryInfo
    load: float
    lastTraining: Optional[str] = None

def get_sensor_id(signal: Signal) -> str:
    """Extrae id de sensor usando el patrón DES_x_y."""
    match = re.match(PATTERN, signal.fileName)
    if not match:
        raise ValueError(f"Nombre de fichero no válido: {signal.fileName}")
    return match.group(2)


def build_scalers(discharges: List[Discharge]):
    """Calcula StandardScaler global para cada feature."""
    sensor_ids = [get_sensor_id(s) for s in discharges[0].signals]
    scalers: Dict[str, StandardScaler] = {}
    for sid in sensor_ids:
        all_values = np.concatenate([
            np.asarray([s.values for s in d.signals if get_sensor_id(s) == sid], dtype=np.float32).ravel()
            for d in discharges
        ]).reshape(-1, 1)
        scaler = StandardScaler().fit(all_values)
        scalers[sid] = scaler
    return scalers


def prepare_windows(discharges: List[Discharge], scalers: Dict[str, StandardScaler]):
    X, y = [], []
    for d in discharges:
        # Normaliza cada señal
        norm_signals = []
        for s in d.signals:
            arr = np.asarray(s.values, dtype=np.float32).reshape(-1, 1)
            arr = scalers[get_sensor_id(s)].transform(arr).ravel()
            # Clipping p99 para robustez
            p99 = np.percentile(arr, 99)
            arr = np.clip(arr, -p99, p99)
            norm_signals.append(arr)
        seq_len = len(norm_signals[0])
        # Sliding windows
        for start in range(0, seq_len - WINDOW + 1, STRIDE):
            window = [sig[start:start+WINDOW] for sig in norm_signals]  # -> list[7][WINDOW]
            X.append(window)
            y.append(1 if d.anomalyTime is not None else 0)
    X = np.transpose(np.array(X, dtype=np.float32), (0, 2, 1))  # (samples, time, features)
    y = np.array(y, dtype=np.float32)
    return X, y

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DilatedCNN(nn.Module):
    def __init__(self, in_channels: int = 7):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32]
        channels = [64, 64, 128, 128, 256, 256]
        layers = []
        prev_c = in_channels
        for d, c in zip(dilations, channels):
            layers.append(nn.Conv1d(prev_c, c, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU())
            prev_c = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev_c, 1)

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)          # -> (B, F, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

class Attention(nn.Module):
    """Simple additive attention layer."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inputs):  # inputs: (B, T, H)
        u = torch.tanh(self.W(inputs))
        scores = self.v(u).squeeze(-1)    # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(inputs * weights, dim=1)

class BiLSTMAttn(nn.Module):
    def __init__(self, hidden=160):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.attn = Attention(hidden*2)
        self.fc = nn.Linear(hidden*2, 1)

    def forward(self, x):  # (B, T, F)
        out, _ = self.lstm(x)
        context = self.attn(out)
        return self.fc(context).squeeze(-1)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, 1)
        self.pos = nn.Parameter(torch.randn(1, WINDOW, d_model))

    def forward(self, x):
        x = self.input_proj(x) + self.pos[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(1)
        return self.cls(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def train_single(model: nn.Module, dataloader: DataLoader, epochs: int, lr: float):
    model.to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running / len(dataloader.dataset):.4f}")

    return model


class Ensemble(nn.Module):
    def __init__(self, cnn: DilatedCNN, lstm: BiLSTMAttn, trf: TimeSeriesTransformer):
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.trf = trf
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        preds = torch.stack([
            torch.sigmoid(self.cnn(x)),
            torch.sigmoid(self.lstm(x)),
            torch.sigmoid(self.trf(x))
        ], dim=1)  # (B, 3)
        w = torch.softmax(self.weights, dim=0)
        return (preds * w).sum(1)

app = FastAPI(title="Disruption Classifier", version="2.0.0")

start_time = time.time()
last_training_time = None
ensemble: Optional[Ensemble] = None

@app.post("/train", response_model=TrainingResponse)
async def train_api(req: TrainingRequest):
    global ensemble, last_training_time
    tic = time.time()

    opts = req.options or TrainingOptions()
    scaler_map = build_scalers(req.discharges)
    X, y = prepare_windows(req.discharges, scaler_map)

    # Dataset & sampler para mitigar desbalance
    dataset = WindowDataset(X, y)
    class_counts = np.bincount(y.astype(int))
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = class_weights[y.astype(int)]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader = DataLoader(dataset, batch_size=opts.batchSize, sampler=sampler, pin_memory=True)

    logger.info(f"Dataset ventanas: {len(dataset)}; anomaly ratio: {class_counts[1]/len(y):.3f}")

    if opts.modelType in ("cnn", "ensemble"):
        cnn = train_single(DilatedCNN(), loader, opts.epochs, opts.learningRate)
    if opts.modelType in ("lstm", "ensemble"):
        lstm = train_single(BiLSTMAttn(), loader, opts.epochs, opts.learningRate)
    if opts.modelType in ("transformer", "ensemble"):
        trf = train_single(TimeSeriesTransformer(), loader, opts.epochs, opts.learningRate)

    if opts.modelType == "ensemble":
        ensemble = Ensemble(cnn, lstm, trf).to(DEVICE)
        # Freeze individual models
        for p in ensemble.cnn.parameters(): p.requires_grad = False
        for p in ensemble.lstm.parameters(): p.requires_grad = False
        for p in ensemble.trf.parameters(): p.requires_grad = False
        # Fine‑tune only weights
        ensemble.train()
        opt_ens = optim.Adam([ensemble.weights], lr=1e-2)
        for _ in range(20):
            total = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt_ens.zero_grad()
                preds = ensemble(Xb)
                loss = F.binary_cross_entropy(preds, yb)
                loss.backward()
                opt_ens.step()
                total += loss.item() * Xb.size(0)
        torch.save(ensemble.state_dict(), ENSEMBLE_PATH)
        model_name = "ensemble"
    else:
        # Save single model
        path = os.path.join(MODEL_DIR, f"{opts.modelType}.pt")
        torch.save(eval(opts.modelType).state_dict(), path)
        model_name = opts.modelType

    os.makedirs(MODEL_DIR, exist_ok=True)
    last_training_time = datetime.datetime.utcnow().isoformat()

    toc = (time.time() - tic) * 1000
    return TrainingResponse(
        status="success",
        message=f"Modelo {model_name} entrenado correctamente",
        trainingId=f"train_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        metrics=TrainingMetrics(),  # Métricas reales se pueden añadir vía MLflow / callback.
        executionTimeMs=toc,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_api(req: PredictionRequest):
    if ensemble is None and not os.path.exists(ENSEMBLE_PATH):
        raise HTTPException(status_code=400, detail="Modelo no entrenado. Llama primero /train")

    # Lazy load
    global ensemble
    if ensemble is None:
        cnn = DilatedCNN().to(DEVICE)
        lstm = BiLSTMAttn().to(DEVICE)
        trf = TimeSeriesTransformer().to(DEVICE)
        ensemble = Ensemble(cnn, lstm, trf).to(DEVICE)
        ensemble.load_state_dict(torch.load(ENSEMBLE_PATH, map_location=DEVICE))
        ensemble.eval()

    tic = time.time()
    scaler_map = build_scalers(req.discharges)
    X, _ = prepare_windows(req.discharges, scaler_map)
    with torch.no_grad():
        preds = torch.sigmoid(ensemble(torch.from_numpy(X).to(DEVICE))).cpu().numpy()
    mean_score = preds.mean()

    toc = (time.time() - tic) * 1000
    return PredictionResponse(
        prediction=int(mean_score > 0.5),
        confidence=float(mean_score if mean_score > 0.5 else 1 - mean_score),
        executionTimeMs=toc,
        model="ensemble",
        details={"windows": len(preds)}
    )

import psutil

@app.get("/health", response_model=HealthCheckResponse)
async def health_api():
    mem = psutil.virtual_memory()
    return HealthCheckResponse(
        status="online" if os.path.exists(ENSEMBLE_PATH) else "degraded",
        version="2.0.0",
        uptime=time.time() - start_time,
        memory=MemoryInfo(total=mem.total / 1_048_576, used=mem.used / 1_048_576),
        load=psutil.cpu_percent() / 100,
        lastTraining=last_training_time,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True,
                limit_concurrency=50, limit_max_requests=20_000, timeout_keep_alive=120)
