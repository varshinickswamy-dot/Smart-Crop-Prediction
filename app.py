import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_CANDIDATES = [BASE_DIR / "datasets", BASE_DIR.parent / "datasets"]
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

def find_asset(filename):
    candidates = [MODEL_DIR / filename, BASE_DIR / filename]
    for path in candidates:
        if path.is_file():
            return path
    return None

def load_joblib(filename):
    path = find_asset(filename)
    if path is None:
        app.logger.warning("Model asset not found: %s", filename)
        return None
    try:
        return joblib.load(path)
    except Exception:
        app.logger.exception("Could not load model asset: %s", path)
        return None

def load_disease_model():
    path = find_asset("disease_model.keras")
    if path is None:
        app.logger.warning("Disease model not found (models/disease_model.keras or project root)")
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except Exception:
        app.logger.exception("Could not load disease model: %s", path)
        return None

disease_model = load_disease_model()
crop_model = load_joblib("crop_model.pkl")
yield_model = load_joblib("yield_model.pkl")
area_enc = load_joblib("area_encoder.pkl")
crop_enc = load_joblib("crop_encoder.pkl")

labels_path = find_asset("disease_labels.json")
try:
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as label_file:
            disease_labels = json.load(label_file)
    else:
        disease_labels = []
except Exception:
    app.logger.exception("Could not load disease labels")
    disease_labels = []

def missing_asset_response(feature):
    return jsonify({
        "error": f"{feature} is not configured yet. Add its trained model file to the models/ folder and redeploy."
    }), 503

def find_yield_dataset():
    for folder in DATASET_CANDIDATES:
        candidate = folder / "yield.csv"
        if candidate.is_file():
            return candidate
    return None

def preprocess_image(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Smart Crop Prediction",
        "features": {
            "crop_prediction": crop_model is not None,
            "disease_detection": disease_model is not None,
            "yield_prediction": yield_model is not None and area_enc is not None and crop_enc is not None,
            "yield_options": find_yield_dataset() is not None
        }
    })

@app.route("/get-yield-options", methods=["GET"])
def get_yield_options():
    dataset_path = find_yield_dataset()
    if dataset_path is None:
        return jsonify({"error": "yield.csv was not found. Place it in the datasets/ folder."}), 503
    try:
        df = pd.read_csv(dataset_path)
        required = {"Area", "Item", "Year"}
        if not required.issubset(df.columns):
            return jsonify({"error": "yield.csv is missing required columns: Area, Item, Year."}), 500
        return jsonify({
            "areas": sorted(df["Area"].dropna().astype(str).unique().tolist()),
            "crops": sorted(df["Item"].dropna().astype(str).unique().tolist()),
            "years": sorted(int(year) for year in pd.to_numeric(df["Year"], errors="coerce").dropna().unique())
        })
    except Exception:
        app.logger.exception("Could not read yield dataset")
        return jsonify({"error": "Unable to read yield dataset."}), 500

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if disease_model is None or not disease_labels:
        return missing_asset_response("Disease detection")
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image using the image field."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No image selected."}), 400
    extension = secure_filename(file.filename).rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Unsupported image format. Use PNG, JPG, JPEG or WEBP."}), 400
    try:
        image = preprocess_image(file.read())
        prediction = disease_model.predict(image, verbose=0)
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        if isinstance(disease_labels, dict):
            disease_name = disease_labels.get(str(index), disease_labels.get(index, f"Class {index}"))
        else:
            disease_name = disease_labels[index]
        disease_name = str(disease_name).replace("___", " - ").replace("_", " ")
        return jsonify({"disease": disease_name, "confidence": round(confidence, 2)})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Disease prediction failed")
        return jsonify({"error": "Disease prediction failed. Please try another image."}), 500

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    if crop_model is None:
        return missing_asset_response("Crop recommendation")
    data = request.get_json(silent=True) or {}
    fields = ["N", "P", "K", "temp", "humidity", "ph", "rainfall"]
    try:
        values = [float(data[field]) for field in fields]
        if not np.all(np.isfinite(values)):
            raise ValueError
        prediction = crop_model.predict([values])[0]
        return jsonify({"crop": str(prediction)})
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide valid numeric values for N, P, K, temp, humidity, ph and rainfall."}), 400
    except Exception:
        app.logger.exception("Crop prediction failed")
        return jsonify({"error": "Crop prediction failed."}), 500

@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    if yield_model is None or area_enc is None or crop_enc is None:
        return missing_asset_response("Yield prediction")
    data = request.get_json(silent=True) or {}
    try:
        area = str(data["area"])
        crop = str(data["crop"])
        year = int(data["year"])
        area_value = int(area_enc.transform([area])[0])
        crop_value = int(crop_enc.transform([crop])[0])
        result = float(yield_model.predict([[area_value, crop_value, year]])[0])
        if not np.isfinite(result):
            raise ValueError("The model returned an invalid result.")
        return jsonify({"yield": round(result, 2)})
    except KeyError:
        return jsonify({"error": "Provide area, crop and year."}), 400
    except ValueError as error:
        return jsonify({"error": str(error) or "Unknown area or crop. Choose a value from the available options."}), 400
    except Exception:
        app.logger.exception("Yield prediction failed")
        return jsonify({"error": "Yield prediction failed."}), 500

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 10 MB."}), 413

@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "The requested endpoint was not found."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
