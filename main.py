from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests

app = Flask(__name__)

# URLs Model di Google Cloud Storage
MODEL_URL = "https://storage.googleapis.com/byetrashmodel/model_weights.weights.h5"
JSON_URL = "https://storage.googleapis.com/byetrashmodel/model.json"

# Unduh model jika belum ada secara lokal
LOCAL_MODEL_PATH = "model_weights.weights.h5"
LOCAL_JSON_PATH = "model.json"

if not os.path.exists(LOCAL_JSON_PATH):
    print("Downloading model JSON...")
    response = requests.get(JSON_URL)
    with open(LOCAL_JSON_PATH, "wb") as f:
        f.write(response.content)

if not os.path.exists(LOCAL_MODEL_PATH):
    print("Downloading model weights...")
    response = requests.get(MODEL_URL)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load model
with open(LOCAL_JSON_PATH, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(LOCAL_MODEL_PATH)

# Labels (ubah sesuai kategori Anda)
LABELS = ["non-organik", "berbahaya", "organik"]

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)
    return image

# API Route: Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Recycling Prediction API!"})

# API Route: Predict
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Preprocess and predict
    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_label = LABELS[np.argmax(prediction)]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)  # Clean up uploaded file

    return jsonify({"prediction": predicted_label, "confidence": float(np.max(prediction))})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
