import dis
from gc import callbacks
from random import shuffle
import re
from tabnanny import verbose
from turtle import mode
from fastapi import FastAPI, HTTPException, Request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Masking, GlobalMaxPooling1D, Flatten, 
    BatchNormalization, Input, Conv1D, Activation, concatenate
    )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import test
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
from signals import Signal as InternalSignal, Discharge as InternalDischarge, DisruptionClass, get_X_y, get_signal_type, pad

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


def is_anomaly(discharge: Discharge) -> bool:
    """Determine if a discharge has an anomaly based on anomalyTime"""
    return discharge.anomalyTime is not None

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    global model, last_training_time
    
    start_execution = time.time()
    
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
                    disruption_class=DisruptionClass.Anomaly if discharge.anomalyTime else DisruptionClass.Normal
                ))
    
        discharges.append(
            InternalDischarge(
                signals=signals, 
                disruption_class=DisruptionClass.Anomaly if discharge.anomalyTime else DisruptionClass.Normal
            )
        )

    # Fill with zeros to make all discharges the same length
    # discharges = pad(discharges)

    try:
        # Configuración: Combinacion de capas CNN 1D para extraer patrones locales y LSTM para capturar dependencias temporales largas
        # 1. 2 bloques consecurivos de capas convolucionales 1D con filtros pequeños (128 a 256 filtros de tamaño 3 a 5), cada uno seguido de una capa de Batch Normalization y una capa de activación ReLU. Estas capas aprenden a filtrar ruido y extraer caracteristicas discriminativas a corto plazo.
        # 2. Una capa LSTM con 64 a 128 unidades, aplicadas sobre las caracteristicas extraidas por las capas convolucionales. Usar `recurrent_dropout` y `dropout` para regularizar. El LSTM aprende patrones temporales a largo plazo.
        # 3. Concatenacion de las salidas de la capa LSTM con las salidas de las capas convolucionales. Esto permite al modelo aprender tanto patrones locales como temporales.
        # 4. Capa de salida: una capa densa con activacion sigmoide para clasificar la salida como normal o anomalia. Como funcion de perdida, usar `binary_crossentropy` y como optimizador `adam` con un learning rate de 0.001.

        # El esquema general es: Input (tiempo x 7 señales) -> CNN1D x2 -> LSTM -> Concatenacion -> Dense (1,sigmoide)
        # Otras opciones son: Usar 3 capas convolucionales 1D y usar LSTM bidireccional.

        # Como tenemos pocas muestras, es critico evitar el sobreajuste. Para ello, usar Dropout y Batch Normalization. Ademas, usar Early Stopping para detener el entrenamiento si la perdida de validacion no mejora durante un numero determinado de epocas.
        
        # Define LSTM model
        model = Sequential()
        # La forma ("shape") de la entrada será (num_descargas, max_signal_length, 7):
        # - num_descargas: tipicamente 6, aunque decide el orquestador
        # - max_signal_length: es el tamaño maximo de las señales, debido a que hemos rellenado con ceros las señales mas cortas
        # - 7: el número de señales (features) que tenemos por descarga
        
        # 1. Definicion de la entrada
        inputs = Input(shape=(None, 7), name="input_layer")
        x = Masking(mask_value=0.0)(inputs)  # Masking layer to ignore padded values

        # 2. Capas convolucionales 1D
        cnn = Conv1D(filters=128, kernel_size=3, padding='same', name="conv1d_layer_1")(x)
        cnn = BatchNormalization(name="bn1")(cnn)
        cnn = Activation('relu', name="act1")(cnn)

        cnn = Conv1D(filters=128, kernel_size=3, padding='same', name="conv1d_layer_2")(cnn)
        cnn = BatchNormalization(name="bn2")(cnn)
        cnn = Activation('relu', name="act2")(cnn)

        # 3. Resumen de la rama CNN
        cnn_branch = GlobalMaxPooling1D(name="global_max_pooling")(cnn)
        
        # 4. Capa LSTM sobre la salida de la CNN
        lstm = LSTM(
            units=64,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_layer"
        )(cnn)
        

        # 5. Concatenar la salida de la LSTM con la salida de la CNN
        merged = concatenate([cnn_branch, lstm], name="concat")

        # 6. Capa de salida
        outputs = Dense(1, activation='sigmoid', name="output_layer")(merged)

        # 7. Contruccion y compilacion del modelo
        model = Model(inputs, outputs, name="CNN_LSTM_Model")
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

        model.summary()
        
        # 8. Entrenamiento del modelo
        X_train, y_train = get_X_y(discharges)
        
        # X_train no tiene la forma (num_descargas, max_signal_length, 7), sino que es del tipo (num_descargas, 7, max_signal_length). Transponer.
        X_train = np.array([np.array(signal).T for signal in X_train])
        y_train = np.array(y_train)
        
        max_length = max(len(seq) for seq in X_train)
                
        X_train = pad_sequences(
            X_train, 
            maxlen=max_length, 
            padding='post', 
            dtype='float32',
            value=0.0
        )
        # print(f"X_train shape: {X_train.shape}")
        # Callbacks para regularizacion y ajuste de learning rate
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=2,
            callbacks=callbacks,
            shuffle=True,
        )
        print(f"Training history: {history.history}")
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
            ).model_dump()
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
                        signal_type=get_signal_type(get_sensor_id(signal.fileName)),
                    ))
        
            discharges.append(
                InternalDischarge(
                    signals=signals, 
                )
            )

        X_pred, _ = get_X_y(discharges)
        max_length = max(len(seq) for seq in X_pred)
        X_pred = pad_sequences(
            X_pred, maxlen=max_length, padding='post', dtype='float32', value=0.0
        )
        # Make predictions
        predictions = model.predict(X_pred, batch_size=2)
        predictions = np.where(predictions > 0.5, 1, 0).flatten()
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

