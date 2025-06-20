# py_lstm
LSTM model for the outlier Classifier.

This service exposes a REST API compatible with the Outlier protocol. Key endpoints:

- `GET /health` – basic node health information.
- `POST /train` and `POST /train/{ordinal}` – training workflow where discharges
  are pushed sequentially.
- `POST /predict` – run a prediction on a single discharge.
