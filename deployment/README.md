# Iris Species Classifier API

## Overview
This API serves a Random Forest model trained on the classic Iris dataset. It provides predictions for Iris species (Setosa, Versicolor, Virginica) based on four morphological features: sepal length, sepal width, petal length, and petal width.

## How to Run
1. Install dependencies:
   ```bash
   pip install flask joblib scikit-learn numpy
   ```
2. Ensure `model.joblib` and `target_names.joblib` are in the root directory.
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. The API will be available at `http://localhost:5000`.

## API Specification

### 1. Health Check
- **Endpoint**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy"}`

### 2. Single Prediction
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Request Format**:
  ```json
  {
    "features": [sepal_length, sepal_width, petal_length, petal_width]
  }
  ```
- **Response Format**:
  ```json
  {
    "predicted_class": "species_name",
    "probabilities": {
      "setosa": 0.0,
      "versicolor": 1.0,
      "virginica": 0.0
    }
  }
  ```

### 3. Batch Prediction
- **Endpoint**: `/predict_batch`
- **Method**: `POST`
- **Request Format**:
  ```json
  {
    "samples": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 2.9, 4.3, 1.3]
    ]
  }
  ```
- **Response Format**:
  ```json
  {
    "predictions": ["setosa", "versicolor"]
  }
  ```

## Example Requests

### Python `requests`
```python
import requests

# Single prediction
sample = [5.1, 3.5, 1.4, 0.2]
response = requests.post("http://localhost:5000/predict", json={"features": sample})
print(response.json())

# Batch prediction
samples = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
response = requests.post("http://localhost:5000/predict_batch", json={"samples": samples})
print(response.json())
```

### curl
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"
```
