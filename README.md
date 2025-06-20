# py_lstm
LSTM model for the outlier classifier. Implements the node protocol used by the orchestrator to manage training and prediction.

### Endpoints

- `GET /health` – reports node status.
- `POST /train` – initiates a training session.
- `POST /train/{ordinal}` – streams discharges for training.
- `POST /predict` – predicts disruption for a single discharge.
