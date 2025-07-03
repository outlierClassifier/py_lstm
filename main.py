import os
import re
from fastapi import FastAPI, HTTPException, Request
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Masking, GlobalMaxPooling1D, Flatten, 
    BatchNormalization, Input, Conv1D, Activation, concatenate,
    MaxPooling1D, SpatialDropout1D, Attention, Bidirectional
    )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import uvicorn
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import datetime
import logging
from signals import Signal as InternalSignal, Discharge as InternalDischarge, DisruptionClass, get_X_y, get_signal_type, generate_more_discharges, pad

PATTERN = "DES_(\\d+)_(\\d+)"
WINDOW_SIZE = 500
OVERLAP = 0.2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
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

class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Initialize FastAPI application
app = FastAPI(
    title="LSTM Anomaly Detection API",
    description="API for anomaly detection using LSTM models",
    version="1.0.0"
)

# Global variables
MODEL_PATH = "lstm_model.keras"
start_time = time.time()
last_training_time = None
model = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        logger.info("Existing model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def get_sensor_id(signal: Signal) -> str:
    """Extrae id de sensor usando el patrón DES_x_y."""
    match = re.match(PATTERN, signal.fileName)
    if match:
        return match.group(2)
    else:
        raise ValueError(f"Invalid signal file name format: {signal.fileName}")

def focal_loss(gamma=2.0, alpha=0.25):
    def f1(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * tf.pow((1 - p_t), gamma) * bce
    return f1

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    global MODEL_PATH
    start_time = time.time()

    # 1) Parse discharges into InternalDischarge
    internal = []
    for d in request.discharges:
        signals = [InternalSignal(
            label=s.fileName,
            times=s.times or d.times,
            values=s.values,
            signal_type=get_signal_type(get_sensor_id(s)),
            disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)
        ) for s in d.signals]
        internal.append(InternalDischarge(
            signals=signals,
            disruption_class=(DisruptionClass.Anomaly if d.anomalyTime else DisruptionClass.Normal)
        ))

    # 2) Optional minimal augmentation
    if len(internal) < 10:
        augmented = []
        for disc in internal:
            augmented.extend(disc.generate_similar_discharges(1))
        internal += augmented

    # 3) Windowing + strategic sampling
    WINDOW_SIZE = 500
    OVERLAP = 0.3
    SAMPLE_PER_DISCHARGE = 80  # incrementar ventanas totales
    windowed, groups = [], []

    for disc_id, disc in enumerate(internal):
        wins = disc.generate_windows(window_size=WINDOW_SIZE, step=1, overlap=OVERLAP)
        total_wins = len(wins)

        # Si es anomalía, centramos alrededor del punto de disrupción
        if disc.disruption_class == DisruptionClass.Anomaly and total_wins > SAMPLE_PER_DISCHARGE:
            # Tomar mitad de ventanas alrededor de la anomalía
            center = total_wins // 2
            half = SAMPLE_PER_DISCHARGE // 2
            start = max(0, center - half)
            end = min(total_wins, center + half)
            idxs = np.arange(start, end)
            # Si no hay suficientes en el rango, rellenar uniformemente
            if len(idxs) < SAMPLE_PER_DISCHARGE:
                extra = np.setdiff1d(np.arange(total_wins), idxs)
                pick = np.random.choice(extra, SAMPLE_PER_DISCHARGE - len(idxs), replace=False)
                idxs = np.concatenate([idxs, pick])
        elif total_wins > SAMPLE_PER_DISCHARGE:
            # No-disruptiva o pocas ventanas: muestreo uniforme
            idxs = np.linspace(0, total_wins - 1, SAMPLE_PER_DISCHARGE, dtype=int)
        else:
            idxs = np.arange(total_wins)

        sampled_wins = [wins[i] for i in idxs]
        windowed.extend(sampled_wins)
        groups.extend([disc_id] * len(sampled_wins))

    # 4) Build input arrays
    X_list, y_list = get_X_y(windowed)
    X = np.stack([np.array(sig).T for sig in X_list])  # shape (n_windows, time, sensors)
    y = np.array(y_list)
    groups = np.array(groups)

    # 5) Model factory
    def build_model():
        inp = Input(shape=(WINDOW_SIZE, X.shape[-1]))
        x = Masking(mask_value=0.0)(inp)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation="sigmoid")(x)
        m = Model(inp, out)
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

    # 6) Leave-One-Discharge-Out CV
    cv = GroupShuffleSplit(n_splits=8, test_size=0.25, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        print(f"--- Fold {fold}/8 ---")
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=8,
            callbacks=callbacks,
            shuffle=True
        )

        probs = model.predict(X_val).ravel()
        threshold = 0.5
        preds = (probs > threshold).astype(int)
        report = classification_report(y_val, preds, labels=[0,1], target_names=["Normal","Anomaly"], zero_division=0)
        print(f"Fold {fold} report:\n{report}")

    # 7) Save final model
    model.save(MODEL_PATH)
    exec_ms = int((time.time() - start_time) * 1000)
    training_id = f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    return TrainingResponse(
        status="success",
        message="Training completed with anomaly-centered sampling",
        trainingId=training_id,
        metrics=TrainingMetrics(accuracy=None, loss=None, f1Score=None),
        executionTimeMs=exec_ms
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model
    
    start_execution = time.time()
    print(f"Received prediction request with {len(request.discharges)} discharges")
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Model not trained",
                code="MODEL_NOT_FOUND",
                details={"message": "Please train the model first"}
            ).model_dump()
        )
    
    try:
        discharges = []
        for discharge in request.discharges:
            signals = []
            for signal in discharge.signals:
                signals.append(
                    InternalSignal(
                        label=signal.fileName,
                        times=signal.times if signal.times else discharge.times,
                        values=signal.values,
                        signal_type=get_signal_type(get_sensor_id(signal)),
                    ))
        
            discharges.append(
                InternalDischarge(
                    signals=signals, 
                )
            )

        # Generate windowed data
        windowed_discharges = []
        for discharge in discharges:
            windowed_discharges.extend(
                discharge.generate_windows(
                    window_size=WINDOW_SIZE, step=1, overlap=OVERLAP
                )
            )
        discharges = windowed_discharges

        X_pred, _ = get_X_y(discharges)
        X_pred = np.array([np.array(signal).T for signal in X_pred])

        # Make predictions
        predictions = model.predict(X_pred, batch_size=2)
        probs = np.where(predictions > 0.5, 1, 0).flatten()
        print(f"Predictions: {predictions}")

        confidence = np.mean(predictions)
        print(f"Confidence: {confidence}")
        
        execution_time = (time.time() - start_execution) * 1000  # ms
        return PredictionResponse(
            prediction=int(predictions[0]),
            confidence=float(confidence),
            executionTimeMs=execution_time,
            model="lstm",
            details={
                "individualPredictions": predictions.tolist(),
                "individualConfidences": confidence.tolist(),
                "numDischargesProcessed": len(request.discharges),
                "featureImportance": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Prediction failed",
                code="PREDICTION_ERROR",
                details={"message": str(e)}
            ).model_dump()
        )

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
    # Try to load the model
    if os.path.exists(MODEL_PATH):
        try:
            model = pickle.load(open(MODEL_PATH, "rb"))
            logger.info("Existing model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            model = None

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False, 
                limit_concurrency=50, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

