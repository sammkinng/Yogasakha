import onnxruntime as ort
from joblib import load

# Load trained MLP model
scaler = load("model/scaler1.pkl")  # Ensure to load the scaler used for training

onnx_model_path = "model/model.onnx"
session = ort.InferenceSession(onnx_model_path)