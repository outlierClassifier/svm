# svm
SVM classifier implementing the node outlier protocol.

### Endpoints

- `GET /health` – returns model name, uptime and last training timestamp.
- `POST /predict` – submit a discharge and obtain a disruption prediction.
- `POST /train` – begin a new training session specifying the number of
  discharges that will be uploaded.
- `POST /train/{ordinal}` – push each discharge sequentially. When the last
  discharge is received the model is trained automatically.
