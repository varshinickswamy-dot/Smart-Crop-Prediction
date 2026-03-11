from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import joblib
import json
import os
import pandas as pd

# ----------------------------------
# APP INIT
# ----------------------------------

app = Flask(__name__)
CORS(app)

# ----------------------------------
# BASE DIRECTORY
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "..", "datasets")

# ----------------------------------
# LOAD MODELS
# ----------------------------------

disease_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "disease_model.keras")
)

crop_model = joblib.load(
    os.path.join(MODEL_DIR, "crop_model.pkl")
)

yield_model = joblib.load(
    os.path.join(MODEL_DIR, "yield_model.pkl")
)

area_enc = joblib.load(
    os.path.join(MODEL_DIR, "area_encoder.pkl")
)

crop_enc = joblib.load(
    os.path.join(MODEL_DIR, "crop_encoder.pkl")
)

with open(os.path.join(MODEL_DIR, "disease_labels.json"), "r") as f:
    disease_labels = json.load(f)

print("✅ All models loaded successfully")

# ----------------------------------
# IMAGE PREPROCESS
# ----------------------------------

def preprocess(img_bytes):

    img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ----------------------------------
# HOME
# ----------------------------------

@app.route("/")
def home():
    return jsonify({"status": "API Running"})


# ----------------------------------
# GET YIELD DATASET OPTIONS
# ----------------------------------

@app.route("/get-yield-options")
def get_yield_options():

    df = pd.read_csv(os.path.join(DATASET_DIR, "yield.csv"))

    areas = sorted(df["Area"].unique().tolist())
    crops = sorted(df["Item"].unique().tolist())
    years = sorted(df["Year"].unique().tolist())

    return jsonify({
        "areas": areas,
        "crops": crops,
        "years": years
    })


# ----------------------------------
# DISEASE PREDICTION
# ----------------------------------

@app.route("/predict_disease", methods=["POST"])
def disease():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = preprocess(file.read())

    pred = disease_model.predict(img)
    index = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100

    disease_name = disease_labels[str(index)] \
        if isinstance(disease_labels, dict) \
        else disease_labels[index]

    disease_name = disease_name.replace("___", " - ").replace("_", " ")

    return jsonify({
        "disease": disease_name,
        "confidence": round(confidence, 2)
    })


# ----------------------------------
# CROP RECOMMENDATION
# ----------------------------------

@app.route("/predict_crop", methods=["POST"])
def crop():

    d = request.json

    X = [[
        d["N"],
        d["P"],
        d["K"],
        d["temp"],
        d["humidity"],
        d["ph"],
        d["rainfall"]
    ]]

    pred = crop_model.predict(X)[0]

    return jsonify({"crop": pred})


# ----------------------------------
# YIELD PREDICTION
# ----------------------------------

@app.route("/predict_yield", methods=["POST"])
def yield_pred():

    d = request.json

    area = area_enc.transform([d["area"]])[0]
    crop = crop_enc.transform([d["crop"]])[0]
    year = int(d["year"])

    y = yield_model.predict([[area, crop, year]])[0]

    return jsonify({"yield": round(float(y), 2)})


# ----------------------------------
# START SERVER
# ----------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)