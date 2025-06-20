import re
from fastapi import FastAPI, HTTPException, Request
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import uvicorn
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import time
import datetime
import os
import pickle
import logging

PATTERN = "DES_(\\d+)_(\\d+)"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for request/response based on API schemas
class Signal(BaseModel):
    filename: str = Field(alias="fileName")
    values: List[float]

    class Config:
        allow_population_by_field_name = True

class Discharge(BaseModel):
    id: str
    signals: List[Signal]
    times: List[float]
    length: int
    # not part of the official schema but kept for legacy training code
    anomalyTime: Optional[float] = None

class PredictionRequest(BaseModel):
    discharge: Discharge

class PredictionEnum(str, Enum):
    Normal = "Normal"
    Anomaly = "Anomaly"

class PredictionResponse(BaseModel):
    prediction: PredictionEnum
    confidence: float
    executionTimeMs: float
    model: str

class TrainingOptions(BaseModel):
    epochs: Optional[int] = None
    batchSize: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class StartTrainingRequest(BaseModel):
    totalDischarges: int = Field(..., ge=1)
    timeoutSeconds: int = Field(..., ge=1)

class StartTrainingResponse(BaseModel):
    expectedDischarges: int

class DischargeAck(BaseModel):
    ordinal: int
    totalDischarges: int

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

class HealthCheckResponse(BaseModel):
    name: str
    uptime: float
    lastTraining: str

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
SCALER_PATH = "scaler.pkl"
start_time = time.time()
last_training_time = "1970-01-01T00:00:00Z"
model = None
scaler = None
training_session = None

# Load model and scaler if they exist
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        logger.info("Existing model loaded successfully")
        
        if os.path.exists(SCALER_PATH):
            scaler = pickle.load(open(SCALER_PATH, "rb"))
            logger.info("Existing scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def get_sensor_id(signal: Signal) -> str:
    """Extract sensor ID from signal file name"""
    match = re.match(PATTERN, signal.filename)
    if match:
        return match.group(2)
    else:
        raise ValueError(f"Invalid signal file name format: {signal.filename}")

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

        scaler = StandardScaler()
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

def prepare_features_for_lstm(discharges: List[Discharge]):
    """
    Extract and prepare features for LSTM model based on the old model's approach.
    
    Args:
        discharges: List of Discharge objects
        
    Returns:
        X_lstm: Feature matrix reshaped for LSTM
        y: Target labels
        scaler: Fitted StandardScaler for future use
    """
    X = []
    y = []
    
    for discharge in discharges:
        # Create dictionary of signal data by sensor ID
        feature_data = {}
        for signal in discharge.signals:
            sensor_id = get_sensor_id(signal)
            # Convert to pandas Series for easy statistical calculations
            feature_data[int(sensor_id)] = {
                'value': pd.Series(signal.values)
            }
        
        # Extract features from each signal
        feature_vector = []
        for feature_num, feature_data in feature_data.items():
            # Simple features: mean, std, min, max, etc.
            feature_vector.extend([
                feature_data['value'].mean(),
                feature_data['value'].std() if len(feature_data['value']) > 1 else 0,  # Handle single value case
                feature_data['value'].min(),
                feature_data['value'].max(),
                feature_data['value'].skew() if len(feature_data['value']) > 2 else 0,  # Handle small sample case
                feature_data['value'].kurtosis() if len(feature_data['value']) > 3 else 0  # Handle small sample case
            ])
        
        X.append(feature_vector)
        y.append(1 if discharge.anomalyTime is not None else 0)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM: [samples, time steps, features]
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    return X_lstm, y, scaler

def prepare_features_for_prediction(discharges: List[Discharge], scaler: StandardScaler):
    """
    Prepare features for prediction using the old model's approach.
    
    Args:
        discharges: List of Discharge objects
        scaler: Fitted StandardScaler
        
    Returns:
        X_lstm: Feature matrix reshaped for LSTM
    """
    X = []
    
    for discharge in discharges:
        # Create dictionary of signal data by sensor ID
        feature_data = {}
        for signal in discharge.signals:
            sensor_id = get_sensor_id(signal)
            # Convert to pandas Series for easy statistical calculations
            feature_data[int(sensor_id)] = {
                'value': pd.Series(signal.values)
            }
        
        # Extract features from each signal
        feature_vector = []
        for feature_num, feature_data in feature_data.items():
            # Simple features: mean, std, min, max, etc.
            feature_vector.extend([
                feature_data['value'].mean(),
                feature_data['value'].std() if len(feature_data['value']) > 1 else 0,  # Handle single value case
                feature_data['value'].min(),
                feature_data['value'].max(),
                feature_data['value'].skew() if len(feature_data['value']) > 2 else 0,  # Handle small sample case
                feature_data['value'].kurtosis() if len(feature_data['value']) > 3 else 0  # Handle small sample case
            ])
        
        X.append(feature_vector)
    
    # Convert to numpy array
    X = np.array(X)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Reshape for LSTM: [samples, time steps, features]
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    return X_lstm

def train_lstm_model(X_train, y_train, options: Optional[TrainingOptions] = None):
    """
    Train the LSTM model using the old model's architecture.
    
    Args:
        X_train: Training feature matrix (reshaped for LSTM)
        y_train: Training labels
        options: Training options from API request
        
    Returns:
        Trained LSTM model and training metrics
    """
    # Set parameters based on options or defaults
    params = {
        'layer1_units': 64,
        'layer2_units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.004,
        'batch_size': 32,
        'epochs': 150
    }
    
    # Override with user-provided options if available
    if options:
        if options.epochs:
            params['epochs'] = options.epochs
        if options.batchSize:
            params['batch_size'] = options.batchSize
        if options.hyperparameters:
            for key, value in options.hyperparameters.items():
                if key in params:
                    params[key] = value
    
    # Log LSTM parameters
    logger.info("LSTM Network Parameters:")
    logger.info(f"  - layer1_units: {params['layer1_units']}")
    logger.info(f"  - layer2_units: {params['layer2_units']}")
    logger.info(f"  - dropout_rate: {params['dropout_rate']}")
    logger.info(f"  - learning_rate: {params['learning_rate']}")
    logger.info(f"  - batch_size: {params['batch_size']}")
    logger.info(f"  - epochs: {params['epochs']}")
    
    # LSTM model for binary classification
    model = Sequential([
        LSTM(params['layer1_units'], input_shape=(1, X_train.shape[2]), return_sequences=True),
        Dropout(params['dropout_rate']),
        LSTM(params['layer2_units']),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Use Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train, 
        epochs=params['epochs'], 
        batch_size=params['batch_size'], 
        verbose=1, 
        validation_split=0.2
    )
    
    # Extract metrics from training history
    metrics = {
        'accuracy': float(history.history['accuracy'][-1]),
        'loss': float(history.history['loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
        'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None
    }
    
    return model, metrics

async def run_training(request: TrainingRequest) -> TrainingResponse:
    global model, scaler, last_training_time
    
    start_execution = time.time()
    
    try:
        # Prepare features using the old model's approach
        X_train, y_train, new_scaler = prepare_features_for_lstm(request.discharges)
        logger.info(f"Training data shape: {X_train.shape}, Labels: {np.sum(y_train)} anomalies out of {len(y_train)}")
        
        # Train model
        trained_model, metrics = train_lstm_model(X_train, y_train, request.options)
        model = trained_model
        scaler = new_scaler
        
        # Save the model and scaler
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        logger.info(f"Scaler saved to {SCALER_PATH}")
        
        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        # Generate training ID
        training_id = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save the model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")
        last_training_time = datetime.datetime.now().isoformat()

        return TrainingResponse(
            status="success",
            message="Training completed successfully",
            trainingId=training_id,
            metrics=TrainingMetrics(
                accuracy=metrics['accuracy'],
                loss=metrics['loss'],
                f1Score=metrics['accuracy']  # Approximation, replace with actual F1 if available
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


@app.post("/train", response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest):
    global training_session

    if training_session is not None:
        raise HTTPException(status_code=503, detail="Node is busy")

    training_session = {"total": request.totalDischarges, "discharges": []}
    return StartTrainingResponse(expectedDischarges=request.totalDischarges)


@app.post("/train/{ordinal}", response_model=DischargeAck)
async def push_discharge(ordinal: int, discharge: Discharge):
    global training_session

    if training_session is None:
        raise HTTPException(status_code=400, detail="No training session")

    if ordinal != len(training_session["discharges"]) + 1 or ordinal > training_session["total"]:
        raise HTTPException(status_code=400, detail="Invalid ordinal")

    training_session["discharges"].append(discharge)
    ack = DischargeAck(ordinal=ordinal, totalDischarges=training_session["total"])

    if ordinal == training_session["total"]:
        resp = await run_training(TrainingRequest(discharges=training_session["discharges"]))
        training_session = None
        callback = os.getenv("TRAINING_CALLBACK_URL")
        if callback:
            try:
                import httpx
                httpx.post(callback, json=resp.dict())
            except Exception as e:
                logger.error(f"Training callback failed: {e}")

    return ack

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model, scaler

    start_execution = time.time()
    logger.info("Received prediction request")
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Model not trained",
                code="MODEL_NOT_FOUND",
                details={"message": "Please train the model first"}
            ).dict()
        )
    
    try:
        # Prepare features for prediction using the old model's approach
        X_predict = prepare_features_for_prediction([request.discharge], scaler)
        logger.info(f"Prediction data shape: {X_predict.shape}")
        
        # Make predictions
        raw_predictions = model.predict(X_predict)
        avg_prediction = float(np.mean(raw_predictions))
        final_prediction = 1 if avg_prediction > 0.5 else 0
        confidence = avg_prediction if final_prediction == 1 else 1 - avg_prediction
        
        # Calculate execution time
        execution_time = (time.time() - start_execution) * 1000  # ms
        
        return PredictionResponse(
            prediction=PredictionEnum.Anomaly if final_prediction == 1 else PredictionEnum.Normal,
            confidence=float(confidence),
            executionTimeMs=execution_time,
            model="lstm"
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
    return HealthCheckResponse(
        name="lstm",
        uptime=time.time() - start_time,
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
    # Try to load the model and scaler
    if os.path.exists(MODEL_PATH):
        try:
            model = pickle.load(open(MODEL_PATH, "rb"))
            logger.info("Existing model loaded successfully")
            
            if os.path.exists(SCALER_PATH):
                scaler = pickle.load(open(SCALER_PATH, "rb"))
                logger.info("Existing scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            model = None

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, 
                limit_concurrency=50, 
                limit_max_requests=20000,
                timeout_keep_alive=120)

