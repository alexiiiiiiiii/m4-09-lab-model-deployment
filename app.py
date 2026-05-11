from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and target names
model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Validation
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400
    
    features = data["features"]
    
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": " 'features' must be a list of 4 numeric values"}), 400
    
    if not all(isinstance(x, (int, float)) for x in features):
        return jsonify({"error": "All features must be numeric"}), 400
    
    # Prediction
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    prob_dict = {name: float(prob) for name, prob in zip(target_names, probabilities)}
    
    return jsonify({
        "predicted_class": target_names[prediction],
        "probabilities": prob_dict
    }), 200

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    
    # Validation
    if not data or "samples" not in data:
        return jsonify({"error": "Missing 'samples' key in request body"}), 400
    
    samples = data["samples"]
    
    if not isinstance(samples, list):
        return jsonify({"error": "'samples' must be a list of feature arrays"}), 400
    
    for i, sample in enumerate(samples):
        if not isinstance(sample, list) or len(sample) != 4:
             return jsonify({"error": f"Sample at index {i} must be a list of 4 numeric values"}), 400
        if not all(isinstance(x, (int, float)) for x in sample):
             return jsonify({"error": f"All values in sample at index {i} must be numeric"}), 400

    # Prediction
    predictions = model.predict(samples)
    
    results = [target_names[p] for p in predictions]
    
    return jsonify({"predictions": results}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
