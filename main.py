import re
from fastapi import FastAPI, HTTPException, Request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import uvicorn
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import datetime
import os
import pickle
import psutil
import logging

PATTERN = "DES_(\\d+)_(\\d+)"

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
    epochs: Optional[int] = None
    batchSize: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None

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
MODEL_PATH = "lstm_model.h5"
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
    """Extract sensor ID from signal file name"""
    match = re.match(PATTERN, signal.fileName)
    if match:
        return match.group(2)
    else:
        raise ValueError(f"Invalid signal file name format: {signal.fileName}")

def sliding_normalized_windows(discharges: list[Discharge], window_size: int, overlap: float):
    """
    ## Sliding window function
    This function takes a list of Discharge objects, normalizes the signals of the same type
    among all discharges, and creates sliding windows of a specified size with a given overlap.

    :param discharges: List of Discharge objects
    :param window_size: Size of the sliding window
    :param overlap: Overlap between windows (0.0 to 1.0)
    :return: Tuple of numpy arrays (X, y) where X is the input data and y is the labels
    """
    step = int(window_size * (1 - overlap))

    sensor_ids = [get_sensor_id(s) for s in discharges[0].signals]

    scalers = {}
    for sid in sensor_ids:
        all_values = np.concatenate([
            np.asarray(
                [s.values for s in d.signals if get_sensor_id(s) == sid],
                dtype=np.float32
            ).ravel()
            for d in discharges
        ]).reshape(-1, 1)

        scaler = MinMaxScaler()
        scaler.fit(all_values)
        scalers[sid] = scaler

    norm_discharge = []
    for d in discharges:
        norm_signals = []
        for s in d.signals:
            arr = np.asarray(s.values).reshape(-1, 1)
            arr_norm = scalers[get_sensor_id(s)].transform(arr).ravel()
            norm_signals.append({
                'id': get_sensor_id(s),
                'values': arr_norm,
            })
        norm_discharge.append({
            'signals': norm_signals,
            'label': 1 if d.anomalyTime else 0,
        })
    
    X, y = [], []
    for d in norm_discharge:
        signals = d['signals']
        label = d['label']
        lenght = len(signals[0]['values'])
        assert all(len(s['values']) == lenght for s in signals), "All signals must have the same length"

        for i in range(0, lenght - window_size + 1, step):
            window = [s['values'][i:i + window_size] for s in signals]
            X.append(window)
            y.append(label)

    X = np.array(X)
    X = np.transpose(X, (0, 2, 1))  # Change shape to (samples, time steps, features)
    return X, np.array(y)

def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    global model, last_training_time
    
    start_execution = time.time()
    
    # Idea general
    # 1. Escalar las 7 features de cada discharge (normalizar) con MinMaxScaler
    # 2. Crear ventanas deslizantes de 500 muestras
    # 3. Crear un array de numpy con las ventanas y otro con los labels (1 si es anomalia, 0 si no).
    #    Todas las ventanas de la misma descarga tienen el mismo label.
    # 4. Los datos que tendremos son:
    #    - 7 features (las 7 señales) por ventana
    #    - Unas 10k / tamano de ventana (500) = 20 ventanas por descarga
    #    - 6 descargas por entrenamiento: 120 ventanas
    # 5. Arquitectura LSTM:
    #    - Tenemos pocas ventanas -> Probablemente una capa
    #    - Units: 32 o 64 (no muchas)
    #    - Dropout: 0.2 o 0.3 (no muchas ventanas)
    #    - Dense: 1 (sigmoid) -> 0 o 1 (anomalía o no)
    #    - Optimizer: Adam (learning rate 0.001)
    #    - Epochs: 200
    #    - Batch size: 32 - 64
    #    - Loss: binary_crossentropy

    try:
        X, y = sliding_normalized_windows(request.discharges, window_size=500, overlap=0.5)
        logger.info(f"Training data shape: {X.shape}, Labels: {np.sum(y)} anomalies out of {len(y)}")
        
        # Define LSTM model
        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X, y, epochs=200, batch_size=32, verbose=1)

        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        # Generate training ID
        training_id = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        return TrainingResponse(
            status="success",
            message="Training completed successfully",
            trainingId=training_id,
            metrics=TrainingMetrics(
                accuracy=0.85,  # Placeholder for actual accuracy
                f1Score=0.8,    # Placeholder for actual F1 score
                loss=0.1,       # Placeholder for actual loss
            ),
            executionTimeMs=execution_time
        )
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                error="Training failed",
                code="TRAINING_ERROR",
                details={"message": str(e)}
            ).dict()
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
            ).dict()
        )
    
    try:
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        X, _ = sliding_normalized_windows(request.discharges, window_size=500, overlap=0.5)
        logger.info(f"Prediction data shape: {X.shape}")
    
        predictions = []
        for i in range(X.shape[0]):
            prediction = model.predict(X[i:i+1])
            predictions.append(prediction[0][0])

        prediction = 1 if np.mean(predictions) > 0.5 else 0
        confidence = np.mean(predictions) if prediction == 1 else 1 - np.mean(predictions)

        return PredictionResponse(
            prediction=prediction,
            confidence=float(confidence),
            executionTimeMs=execution_time,
            model="lstm",
            details={
                "individualPredictions": [],
                "individualConfidences": 0.4,
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
            ).dict()
        )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    # Get memory information
    mem = psutil.virtual_memory()
    
    return HealthCheckResponse(
        status="online" if model is not None else "degraded",
        version="1.0.0",
        uptime=time.time() - start_time,
        memory=MemoryInfo(
            total=mem.total / (1024*1024),  # Convert to MB
            used=mem.used / (1024*1024)
        ),
        load=psutil.cpu_percent() / 100,
        lastTraining=last_training_time
    )

# Custom middleware to handle large request JSON payloads
@app.middleware("http")
async def increase_json_size_limit(request: Request, call_next):
    # Increase JSON size limit for this specific request
    # Default is 1MB, we're setting it to 64MB
    request.app.state.json_size_limit = 64 * 1024 * 1024  # 64MB
    response = await call_next(request)
    return response

if __name__ == "__main__":
    # discharge1 = Discharge(
    #     id="DES_1",
    #     signals=[
    #         Signal(fileName="DES_1_1", values=[1, 2, 3], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_2", values=[6, 5, 4], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_3", values=[6, 5, 4], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_4", values=[6, 5, 4], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_5", values=[6, 5, 4], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_6", values=[6, 5, 4], times=[1, 2, 3]),
    #         Signal(fileName="DES_1_7", values=[7, 8, 9], times=[1, 2, 3]),
    #     ],
    #     anomalyTime=1.5
    # )
    # discharge2 = Discharge(
    #     id="DES_2",
    #     signals=[
    #         Signal(fileName="DES_2_1", values=[3, 4, 5], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_2", values=[3, 2, 1], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_3", values=[4, 5, 6], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_4", values=[7, 8, 9], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_5", values=[7, 8, 9], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_6", values=[7, 8, 9], times=[1, 2, 3]),
    #         Signal(fileName="DES_2_7", values=[7, 8, 9], times=[1, 2, 3]),
    #     ],
    #     anomalyTime=None
    # )
    # discharges = [discharge1, discharge2]
    # X,y = sliding_window(discharges, window_size=2, overlap=0.5)

    # print(f"X: {X}, y: {y}")

    # Try to load the model
    if os.path.exists(MODEL_PATH):
        try:
            model = pickle.load(open(MODEL_PATH, "rb"))
            logger.info("Existing model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            model = None

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, 
                limit_concurrency=50, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

