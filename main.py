import os
import re
import time
import datetime
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

LOGGER_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGER_FMT)
logger = logging.getLogger("disruption-classifier")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR = "artifacts"
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble.pt")
FORECASTER_PATH = os.path.join(MODEL_DIR, "forecaster.pt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "tau.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 16  # Number of time steps per window
STRIDE = 0        # Overlap disabled

MODEL_NAME = "LSTM"
PATTERN = r"DES_(\d+)_(\d+)"

A_MINOR = 0.95  # [m] Minor radius for Greenwald limit on JET
EPOCHS_DEFAULT = 30
LEARNING_RATE_DEFAULT = 1e-3

EPS = 1e-12    # avoid division by zero

# ratios caps to avoid explosions before log
CAP_RAD_FRAC      = 1e6       # P_rad / P_in
CAP_GREENWALD_FR  = 10.0      # n_e / n_G (physically ~O(1), but we leave some margin)
CAP_BETA_LOSS     = 1e6       # |dW/dt| / P_in
CAP_LM_NORM       = 1e6       # |LM| / I_p
CAP_LI_NORM       = 1e6       # l_i / I_p
CAP_LOG_FEATURE   = 50.0   

class Signal(BaseModel):
    filename: str
    values: List[float]

class Discharge(BaseModel):
    id: str
    signals: List[Signal]
    times: List[float]
    length: int
    anomalyTime: Optional[float] = None

    @field_validator("signals", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> List[Signal] | Any:
        if isinstance(v, dict):
            return [v]
        return v

class StartTrainingRequest(BaseModel):
    totalDischarges: int = Field(..., ge=1)
    timeoutSeconds: int = Field(..., ge=1)

class StartTrainingResponse(BaseModel):
    expectedDischarges: int

class DischargeAck(BaseModel):
    ordinal: int
    totalDischarges: int

class TrainingMetrics(BaseModel):
    accuracy: float
    loss: float
    f1Score: float

class TrainingResponse(BaseModel):
    status: str
    message: str
    trainingId: str
    metrics: TrainingMetrics
    executionTimeMs: float

class WindowProperties(BaseModel):
    featureValues: List[float] = Field(..., min_items=1)
    prediction: str = Field(..., pattern=r'^(Anomaly|Normal)$')
    justification: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str
    windowSize: int = WINDOW_SIZE
    windows: List[WindowProperties]

class HealthCheckResponse(BaseModel):
    name: str = MODEL_NAME
    uptime: float
    lastTraining: str

class SeqDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray]):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx]

def pad_collate_forecast(batch):
    # batch: array list (T_i, D). Contructs X=(T_i-1), Y=(T_i-1) with next-step.
    Ds = [torch.from_numpy(b) for b in batch]
    lens = [d.size(0) for d in Ds]
    # drop sequences of length 1 (no next-step)
    keep = [i for i, L in enumerate(lens) if L >= 2]
    Ds = [Ds[i] for i in keep]; lens = [lens[i] for i in keep]
    Tm1 = max(L-1 for L in lens)
    D = Ds[0].size(1)
    X = torch.zeros(len(Ds), Tm1, D)
    Y = torch.zeros(len(Ds), Tm1, D)
    Lm1 = torch.zeros(len(Ds), dtype=torch.long)
    for i, d in enumerate(Ds):
        L = d.size(0)
        x = d[:L-1, :]
        y = d[1:L, :]
        X[i, :L-1] = x
        Y[i, :L-1] = y
        Lm1[i] = L-1
    return X, Y, Lm1

def get_sensor_id(signal: Signal) -> str:
    """Extracts sensor ID using the pattern DES_x_y."""
    match = re.match(PATTERN, signal.filename)
    if not match:
        raise ValueError(f"Invalid filename: {signal.filename}")
    return match.group(2)

def _safe_div(a, b):
    return 0.0 if (b == 0 or not np.isfinite(a) or not np.isfinite(b)) else (a / b)

def _safe_ratio(a: float, b: float, cap: float, nonneg: bool = False) -> float:
    """Returns (a / max(|b|, EPS)) capped to ±cap (or [0, cap] if nonneg=True)."""
    denom = max(abs(b), EPS)
    r = a / denom
    if nonneg:
        r = max(0.0, r)
        return float(min(r, cap))
    return float(np.clip(r, -cap, cap))

def extract_aux_features(win_raw: np.ndarray,
                         prev_logs: Optional[tuple[float, float]]) -> tuple[np.ndarray, tuple[float, float]]:
    # f64 to avoid overflow
    wr64 = win_raw.astype(np.float64, copy=False)

    mean_vals = np.mean(wr64, axis=1) 
    Ip, LM, LI, NE, dWdt, Prad, Pin = mean_vals
    Ip_MA   = Ip * 1e-6
    ne_1e20 = NE * 1e-20

    rad_frac  = _safe_ratio(Prad, Pin,  CAP_RAD_FRAC, nonneg=True)
    greenwald = _safe_ratio(ne_1e20, Ip_MA/(np.pi*A_MINOR**2 + EPS), CAP_GREENWALD_FR, nonneg=True)
    LM_norm   = _safe_ratio(abs(LM), max(Ip_MA, EPS), CAP_LM_NORM, nonneg=True)
    li_norm   = _safe_ratio(LI,      max(Ip_MA, EPS), CAP_LI_NORM, nonneg=False)
    beta_loss = _safe_ratio(abs(dWdt), max(Pin, EPS), CAP_BETA_LOSS, nonneg=True)

    # avoid heterogeneous values in std
    cross_std = float(np.std(np.log1p(np.abs(mean_vals) + 1e-12)))

    Elog = float(np.log1p(np.mean(wr64**2)))

    L_rad  = float(np.log1p(rad_frac))
    L_gw   = float(np.log1p(greenwald))
    L_lm   = float(np.log1p(LM_norm))
    L_li   = float(np.log1p(abs(li_norm)))
    L_beta = float(np.log1p(beta_loss))

    if prev_logs is None:
        d_rad=d_beta=d_gw=d_lm=d_li = 0.0
    else:
        prev_Lrad, prev_Lbeta, prev_Lgw, prev_Llm, prev_Lli = prev_logs
        d_rad  = float(L_rad  - prev_Lrad)
        d_beta = float(L_beta - prev_Lbeta)
        d_gw   = float(L_gw   - prev_Lgw)
        d_lm   = float(L_lm   - prev_Llm)
        d_li   = float(L_li   - prev_Lli)

    feat64 = np.array([L_rad, L_gw, L_lm, L_li, L_beta, cross_std, Elog, d_rad, d_beta, d_gw, d_lm, d_li], 
                      dtype=np.float64)

    feat64 = np.nan_to_num(feat64, posinf=CAP_LOG_FEATURE, neginf=-CAP_LOG_FEATURE)
    feat64 = np.clip(feat64, -CAP_LOG_FEATURE, CAP_LOG_FEATURE)

    return feat64.astype(np.float32), (L_rad, L_beta, L_gw, L_lm, L_li)

def build_feature_scaler(feat_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    F = np.vstack(feat_list).astype(np.float64, copy=False)   # avoid overflow in std
    mu = np.nanmean(F, axis=0)
    sd = np.nanstd(F,  axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


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

def prepare_feature_sequences(discharges: List[Discharge]) -> tuple[list[np.ndarray], dict]:
    sequences: list[np.ndarray] = []
    all_feats: list[np.ndarray] = []

    dropped_by_shot = 0
    kept_by_shot = 0

    for d in discharges:
        raw = np.stack([np.asarray(s.values, dtype=np.float32) for s in d.signals])  # (7, T)
        T = raw.shape[1]
        feats = []
        prev = None
        local_dropped = 0
        for start in range(0, T - WINDOW_SIZE + 1, max(1, WINDOW_SIZE)):
            win_raw = raw[:, start:start+WINDOW_SIZE]
            if win_raw.shape[1] < WINDOW_SIZE:
                break
            # raw filter
            if not np.isfinite(win_raw).all():
                local_dropped += 1
                continue

            f, next_prev = extract_aux_features(win_raw, prev)
            if not np.isfinite(f).all():
                local_dropped += 1
                continue

            feats.append(f); all_feats.append(f)
            prev = next_prev

        if len(feats) > 0:
            sequences.append(np.stack(feats).astype(np.float32))  # (Nw, D)
            kept_by_shot += len(feats)
        dropped_by_shot += local_dropped

    if len(all_feats) == 0:
        raise RuntimeError("No valid feature windows after sanity filtering (NaN/Inf). Check input signals.")

    # global scaler per feature (ε included at build_feature_scaler)
    mu, sd = build_feature_scaler(all_feats)
    stats = {"mean": mu.tolist(), "std": sd.tolist()}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "feat_stats.json"), "w") as f:
        import json; json.dump(stats, f)

    # normalize sequences
    sequences = [((seq - mu) / sd).astype(np.float32) for seq in sequences]

    try:
        logger.info(f"Kept feature windows: {kept_by_shot}, dropped: {dropped_by_shot}")
    except Exception:
        pass
    return sequences, stats


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
        self.pos = nn.Parameter(torch.randn(1, WINDOW_SIZE, d_model))

    def forward(self, x):
        x = self.input_proj(x) + self.pos[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(1)
        return self.cls(x).squeeze(-1)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=7, hidden=128):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden, input_size)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = torch.zeros(x.size(0), x.size(1), self.encoder.hidden_size, device=x.device)
        out, _ = self.decoder(dec_in, (h, c))
        return self.output(out)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden: int = 192, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden, input_size)

    def forward(self, x):
        # x: (B, T, D) -> yhat: (B, T, D)
        h, _ = self.lstm(x)
        return self.proj(h)


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

def train_autoencoder(model: nn.Module, dataloader: DataLoader, epochs: int, lr: float):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        count = 0
        for X, _, lengths in dataloader:          # ← ahora recibimos 3 elementos
            X = X.to(DEVICE)
            lengths = lengths.to(DEVICE)
            optimizer.zero_grad()
            recon = model(X)                      # (B, T, D)
            err_t = F.smooth_l1_loss(recon, X, reduction='none').mean(dim=2)  # (B, T)
            T = X.size(1)
            mask = (torch.arange(T, device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)).float()
            loss = (err_t * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            running += loss.item()
            count += 1
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running / max(1,count):.6f}")
    model.eval()
    return model

# def compute_threshold(model: nn.Module, dataloader: DataLoader, fa_rate: float = 0.005) -> float:
#     model.eval()
#     errors = []
#     with torch.no_grad():
#         for X, _ in dataloader:
#             X = X.to(DEVICE)
#             recon = model(X)
#             err = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
#             errors.extend(err)
#     tau = float(np.percentile(errors, 100 * (1 - fa_rate)))
#     np.save(THRESHOLD_PATH, tau)
#     print(f"Computed threshold value: {tau}")
#     return tau

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
g_model: Optional[LSTMForecaster] = None
tau_value: Optional[float] = None
expected_discharges: int = 0
buffered_discharges: List[Discharge] = []


def run_training_job(discharges: List[Discharge]):
    """Internal training logic."""
    global g_model, tau_value, last_training_time
    tic = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    sequences, stats = prepare_feature_sequences(discharges)

    # Train only with non-disruptive discharges
    pairs = [(seq, d.id) for seq, d in zip(sequences, discharges) if d.anomalyTime is None]
    if not pairs:
        raise RuntimeError("No non-disruptive discharges to train the forecaster.")

    rng = np.random.RandomState(SEED)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_pairs = [pairs[i][0] for i in idx[:cut]]
    val_pairs   = [pairs[i][0] for i in idx[cut:]]

    # DataLoaders
    ds_tr = SeqDataset(train_pairs)
    ds_va = SeqDataset(val_pairs)

    loader_tr = DataLoader(ds_tr, batch_size=8, shuffle=True,  collate_fn=pad_collate_forecast)
    loader_va = DataLoader(ds_va, batch_size=8, shuffle=False, collate_fn=pad_collate_forecast)

    input_size = train_pairs[0].shape[1]
    model = LSTMForecaster(input_size=input_size, hidden=192, num_layers=1).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE_DEFAULT, weight_decay=1e-2)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, min_lr=1e-5)

    best_val = float('inf'); bad = 0; patience = 5
    for epoch in range(EPOCHS_DEFAULT):
        # ---- train ----
        model.train(); tr_loss = 0.0; nbt = 0
        for X, Y, Lm1 in loader_tr:
            X, Y, Lm1 = X.to(DEVICE), Y.to(DEVICE), Lm1.to(DEVICE)
            opt.zero_grad()
            Yhat = model(X)                                     # (B, Tm1, D)
            err_t = F.smooth_l1_loss(Yhat, Y, reduction='none').mean(dim=2)  # (B, Tm1)
            Tm1 = X.size(1)
            mask = (torch.arange(Tm1, device=DEVICE).unsqueeze(0) < Lm1.unsqueeze(1)).float()
            loss = (err_t * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward(); opt.step()
            tr_loss += loss.item(); nbt += 1

        # ---- val ----
        model.eval(); va_loss = 0.0; nbv = 0
        with torch.no_grad():
            for X, Y, Lm1 in loader_va:
                X, Y, Lm1 = X.to(DEVICE), Y.to(DEVICE), Lm1.to(DEVICE)
                Yhat = model(X)
                err_t = F.smooth_l1_loss(Yhat, Y, reduction='none').mean(dim=2)
                Tm1 = X.size(1)
                mask = (torch.arange(Tm1, device=DEVICE).unsqueeze(0) < Lm1.unsqueeze(1)).float()
                loss = (err_t * mask).sum() / mask.sum().clamp_min(1.0)
                va_loss += loss.item(); nbv += 1
        va_loss /= max(1, nbv); tr_loss /= max(1, nbt)
        logger.info(f"Epoch {epoch+1}/{EPOCHS_DEFAULT} - Train: {tr_loss:.6f} - Val: {va_loss:.6f}")
        sched.step(va_loss)
        if va_loss + 1e-4 < best_val:
            best_val = va_loss; bad = 0
            torch.save(model.state_dict(), FORECASTER_PATH)
        else:
            bad += 1
            if bad >= patience: break

    model.load_state_dict(torch.load(FORECASTER_PATH, map_location=DEVICE))
    model.eval()

    def discharge_p95_forecast(seqs: list[np.ndarray]) -> np.ndarray:
        vals = []
        with torch.no_grad():
            for seq in seqs:
                if seq.shape[0] < 2:
                    continue
                x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(DEVICE)  # (1, T-1, D)
                y = torch.from_numpy(seq[1:]).unsqueeze(0).to(DEVICE)   # (1, T-1, D)
                yhat = model(x)
                e = F.smooth_l1_loss(yhat, y, reduction='none').mean(dim=2).squeeze(0).cpu().numpy()
                vals.append(np.percentile(e, 95))
        return np.array(vals, dtype=np.float32)

    neg_p95 = discharge_p95_forecast(val_pairs)
    finite = np.isfinite(neg_p95)
    if not finite.all():
        neg_p95 = neg_p95[finite]
    if neg_p95.size == 0:
        raise RuntimeError("Empty validation for tau.")

    # light clamping to avoid p99 coinciding with a single extreme
    hi = np.quantile(neg_p95, 0.995) if neg_p95.size > 200 else np.quantile(neg_p95, 0.99)
    neg_p95 = np.clip(neg_p95, None, hi)
    tau = float(np.quantile(neg_p95, 0.99))
    np.save(THRESHOLD_PATH, np.array(tau, dtype=np.float32))

    tau_value = tau
    last_training_time = datetime.datetime.now().isoformat()
    g_model = model
    logger.info(
        f"Training completed in {(time.time() - tic):.2f} s; "
        f"τ(p99 of p95)={tau:.6f} | neg_p95 stats "
        f"min/med/p95/p99=({neg_p95.min():.3f}/{np.median(neg_p95):.3f}/"
        f"{np.percentile(neg_p95,95):.3f}/{np.percentile(neg_p95,99):.3f})"
    )
    print(f"Updated threshold value: {tau}")


@app.post("/train", response_model=StartTrainingResponse)
async def start_training(req: StartTrainingRequest):
    global expected_discharges, buffered_discharges
    if expected_discharges != 0:
        raise HTTPException(status_code=503, detail="Busy")
    expected_discharges = req.totalDischarges
    buffered_discharges = []
    print(f"Training session started: {expected_discharges} discharges expected")
    return StartTrainingResponse(expectedDischarges=expected_discharges)


@app.post("/train/{ordinal}", response_model=DischargeAck)
async def push_discharge(ordinal: int, discharge: Discharge, background_tasks: BackgroundTasks):
    global expected_discharges, buffered_discharges
    if expected_discharges == 0:
        raise HTTPException(status_code=400, detail="Session not started")
    if ordinal < 1 or ordinal > expected_discharges:
        raise HTTPException(status_code=400, detail="Invalid ordinal")
    buffered_discharges.append(discharge)
    ack = DischargeAck(ordinal=ordinal, totalDischarges=expected_discharges)
    if ordinal == expected_discharges:
        background_tasks.add_task(run_training_job, buffered_discharges.copy())
        expected_discharges = 0
        buffered_discharges = []
    return ack


@app.post("/predict", response_model=PredictionResponse)
async def predict_api(discharge: Discharge):
    global g_model, tau_value

    if tau_value is None and os.path.exists(THRESHOLD_PATH):
        tau_value = float(np.load(THRESHOLD_PATH))
        print("pre-computed threshold value loaded")
    if g_model is None or tau_value is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    tic = time.time()

    raw = np.stack([np.asarray(s.values, dtype=np.float32) for s in discharge.signals])  # (7,T)
    feats, prev = [], None
    for start in range(0, raw.shape[1] - WINDOW_SIZE + 1, max(1, WINDOW_SIZE)):
        f, prev = extract_aux_features(raw[:, start:start+WINDOW_SIZE], prev)
        feats.append(f)
    if len(feats) == 0:
        raise HTTPException(status_code=400, detail="No valid windows")

    import json
    stats_path = os.path.join(MODEL_DIR, "feat_stats.json")
    with open(stats_path, "r") as f:
        st = json.load(f)
    mu = np.array(st["mean"], dtype=np.float32); sd = np.array(st["std"], dtype=np.float32)
    seq = ((np.stack(feats).astype(np.float32) - mu) / sd).astype(np.float32)  # (Nw, D)

    if g_model is None and os.path.exists(FORECASTER_PATH):
        g_model = LSTMForecaster(input_size=seq.shape[1], hidden=192, num_layers=1).to(DEVICE)
        g_model.load_state_dict(torch.load(FORECASTER_PATH, map_location=DEVICE))
        g_model.eval()
        print("pre-trained forecaster model loaded")

    if g_model is None:
        raise HTTPException(status_code=503, detail="Model not trained (weights missing)")

    if getattr(g_model.proj, "out_features", None) != seq.shape[1]:
        raise HTTPException(status_code=500, detail="Feature dimension mismatch; retrain forecaster.")

    with torch.no_grad():
        if seq.shape[0] < 2:
            raise HTTPException(status_code=400, detail="Sequence too short for prediction")
        x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(DEVICE)  # (1, T-1, D)
        y = torch.from_numpy(seq[1:]).unsqueeze(0).to(DEVICE)
        yhat = g_model(x)
        err = F.smooth_l1_loss(yhat, y, reduction='none').mean(dim=2).squeeze(0).cpu().numpy()

    shot_p95 = float(np.percentile(err, 95))
    is_anom = shot_p95 >= tau_value
    ratio = shot_p95 / (tau_value + 1e-8)
    conf  = float(np.clip(ratio if is_anom else 1.0 - ratio, 0.0, 1.0))
    err_aligned = np.concatenate([err, err[-1:]], axis=0)
    probs = np.clip(err_aligned / (tau_value + 1e-8), 0.0, 1.0)

    windows = [
        WindowProperties(
            featureValues=seq[i].tolist(),
            prediction=("Anomaly" if probs[i] >= 1.0 else "Normal"),
            justification=float(probs[i])
        ) for i in range(len(probs))
    ]

    exec_ms = (time.time() - tic) * 1000.0
    return PredictionResponse(
        prediction=("Anomaly" if is_anom else "Normal"),
        confidence=conf,
        executionTimeMs=exec_ms,
        model="lstm_forecaster_features",
        windowSize=WINDOW_SIZE,
        windows=windows
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_api():
    return HealthCheckResponse(
        name=MODEL_NAME,
        uptime=time.time() - start_time,
        lastTraining=last_training_time or ""
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002)
