import pandas as pd
import joblib
import os
import json
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------------------------------
# CREATE MODELS FOLDER
# -------------------------------------------------------
os.makedirs("models", exist_ok=True)

# =======================================================
# 1️⃣ DISEASE DETECTION MODEL (HIGH ACCURACY VERSION)
# =======================================================

dataset_path = "../datasets/PlantVillage"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

class_names = list(train_data.class_indices.keys())

with open("models/disease_labels.json", "w") as f:
    json.dump(class_names, f)

print("Number of disease classes:", len(class_names))

# -----------------------
# Transfer Learning Model
# -----------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(len(class_names), activation="softmax")(x)

disease_model = Model(inputs=base_model.input, outputs=predictions)

disease_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("models/best_disease_model.keras", save_best_only=True)
]

history = disease_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

disease_model.save("models/disease_model.keras")
print("Disease model trained and saved")

# =======================================================
# 2️⃣ CROP RECOMMENDATION MODEL
# =======================================================

crop_df = pd.read_csv("../datasets/Crop_recommendation.csv")

X_crop = crop_df.iloc[:, :-1]
y_crop = crop_df.iloc[:, -1]

crop_model = RandomForestClassifier(n_estimators=300)
crop_model.fit(X_crop, y_crop)

joblib.dump(crop_model, "models/crop_model.pkl")
print("Crop model saved")

# =======================================================
# 3️⃣ YIELD PREDICTION MODEL
# =======================================================

yield_df = pd.read_csv("../datasets/yield.csv")

yield_df = yield_df[yield_df["Element"] == "Yield"]
yield_df = yield_df[["Area", "Item", "Year", "Value"]]
yield_df.columns = ["area", "crop", "year", "yield"]

area_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

yield_df["area"] = area_encoder.fit_transform(yield_df["area"])
yield_df["crop"] = crop_encoder.fit_transform(yield_df["crop"])

X_yield = yield_df[["area", "crop", "year"]]
y_yield = yield_df["yield"]

yield_model = RandomForestRegressor(n_estimators=400)
yield_model.fit(X_yield, y_yield)

joblib.dump(yield_model, "models/yield_model.pkl")
joblib.dump(area_encoder, "models/area_encoder.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")

print("Yield model saved")

print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
