# py_lstm
LSTM based models for disruption detection. This repository provides both a
supervised ensemble classifier and an unsupervised autoencoder for anomaly
detection. The HTTP API implements the Node Protocol with a `/train` handshake
followed by `/train/{ordinal}` uploads and a `/predict` endpoint for single
discharge predictions.
