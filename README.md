# Smart Crop Prediction

A Flask-based agricultural decision-support application with crop recommendation, plant disease detection, and crop-yield prediction.

## Features
- Crop recommendation from N, P, K, temperature, humidity, pH and rainfall.
- Plant disease classification from an uploaded leaf image.
- Yield estimation using area, crop and year.
- Responsive web interface and JSON API.
- Health endpoint at `/health`.

## Required model assets

The repository's original training code writes trained assets to `models/`. Place these files in that folder before expecting all prediction features to work:

```
models/
  crop_model.pkl
  yield_model.pkl
  area_encoder.pkl
  crop_encoder.pkl
  disease_model.keras
  disease_labels.json
datasets/
  yield.csv
  Crop_recommendation.csv
  PlantVillage/...
```

For backward compatibility, the app also looks for individual model files in the project root. The current GitHub repository does not include the trained `crop_model.pkl`, `yield_model.pkl`, or `disease_model.keras` files, nor the datasets. Do not commit large datasets or model files unless they fit your repository and hosting limits; use Git LFS or a supported model-artifact store when appropriate.

The web service can start without model files so the deployment and interface can be checked. The relevant prediction endpoints return HTTP 503 with a clear message until their model assets are supplied. The `/health` endpoint reports which features are ready.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## Deploy on Render (free web service)

1. Push this repository to GitHub.
2. In Render, choose **New + → Web Service** and connect this repository.
3. Select the branch `main`.
4. Use:
   - Environment: **Python**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Instance: Free (if available for your account/region)
5. Deploy. Open the service URL and check `/health`.
6. Add the trained model files and `datasets/yield.csv` using a suitable artifact-storage approach, then redeploy.

Render free instances may spin down after inactivity and have limited memory/disk. TensorFlow model loading can exceed free-tier memory; if that happens, use a smaller inference runtime/model or a paid instance. Do not put secrets in source code.

## API endpoints

- `GET /` — responsive application interface
- `GET /health` — service and model readiness
- `GET /get-yield-options` — areas, crops and years from `datasets/yield.csv`
- `POST /predict_crop` — JSON fields: `N`, `P`, `K`, `temp`, `humidity`, `ph`, `rainfall`
- `POST /predict_yield` — JSON fields: `area`, `crop`, `year`
- `POST /predict_disease` — multipart form field: `image`

## Training

The existing `train_models.py` is retained. Run it from the project root only after placing the training datasets in the expected `../datasets/` location, or update the dataset paths for your local dataset location. Training is not part of the Render build process.
