# 🩺 MedicalAIPlatform

AI-assisted screening of medical scans. Upload a scan, get a predicted class with confidence, a risk level, a Grad-CAM heatmap showing what the model looked at, and an optional plain-language report written by Google Gemini.

> **Disclaimer:** This is a student project built for educational purposes. It is **not** a medical device and its output is **not** a medical diagnosis. Always consult a qualified healthcare professional.

---

## Features

- **Three conditions supported**
  | Condition | Scan type | Classes |
  |---|---|---|
  | Brain Tumor | MRI | Glioma, Meningioma, No Tumor, Pituitary |
  | Pneumonia | Chest X-ray | Normal, Pneumonia |
  | Diabetic Retinopathy | Retina image | No DR, Mild, Moderate, Severe, Proliferative DR |
- **Transfer learning** with ResNet50 (ImageNet weights, frozen base, small trainable head)
- **Risk analysis**: each class maps to Low / Moderate / High risk, and confidence is rated High (≥ 90%), Moderate (≥ 70%) or Low. Low-confidence results are always flagged for review
- **Grad-CAM** heatmaps for explainability
- **AI reports** via Google Gemini (retry logic, timeout, and a built-in disclaimer on every report), downloadable as `.txt`
- **History** of predictions and reports stored in SQLite
- **Input handling**: JPG, PNG or PDF (first page is used), 10 MB limit, minimum image size check, sanitized upload names
- **Streamlit** multi-page UI: Home, Prediction, History, About

## How it works

```
Upload (JPG/PNG/PDF) → preprocess (224×224 RGB) → ResNet50 classifier
        → class + confidence → risk mapping → Grad-CAM → (optional) Gemini report
                                   ↓
                         SQLite history (predictions & reports)
```

Each dataset has a `Training` and a `Testing` folder; a stratified ~15% validation set is split off the training images. Training uses class weights for imbalanced data, light augmentation (flip, small rotation, zoom, contrast) on training images only, and Adam (lr = 1e-4) with early stopping, LR reduction on plateau and best-model checkpointing.

## Project structure

```
MedicalAIPlatform/
├── app/
│   ├── Home.py                  # Streamlit entry point
│   ├── ui.py                    # Theme and shared UI components
│   └── pages/                   # Prediction, History, About
├── assets/                      # Static assets
├── config/
│   ├── config.py                # Settings dataclass and project directories
│   ├── configuration.py         # Values read from .env
│   └── constants.py             # Classes, risk mapping, hyperparameters, paths
├── logs/
│   └── medical_ai.log
├── notebooks/                   # Experiments and exploration
├── saved_models/                # Trained .keras models (e.g. pneumonia.keras)
├── src/
│   ├── ai_services/
│   │   ├── ai_client.py         # Gemini client with retries and timeout
│   │   ├── prompt_builder.py
│   │   └── report_analyser.py
│   ├── database/
│   │   ├── db_handler.py        # SQLite connection and table setup
│   │   ├── history_manager.py   # Prediction history
│   │   └── report_storage.py    # Saved AI reports
│   ├── evaluation/
│   │   └── evaluator.py         # Accuracy, precision, recall, F1, confusion matrix
│   ├── inference/
│   │   ├── predictor.py
│   │   └── riskanalyser.py
│   ├── models/
│   │   ├── gradcam.py
│   │   └── model_loader.py
│   ├── pipeline/
│   │   ├── augmentation.py
│   │   ├── input_loader.py      # JPG / PNG / PDF loading
│   │   ├── preprocessing.py
│   │   └── upload_handler.py
│   ├── training/
│   │   ├── callbacks.py
│   │   ├── dataset_loader.py
│   │   ├── model_builder.py
│   │   ├── model_compiler.py
│   │   ├── savemodel.py
│   │   └── trainer.py
│   └── utils/
│       ├── exception.py
│       └── logger.py
├── tests/
│   ├── conftest.py
│   └── test_edge_cases.py
├── .env                         # API keys (not committed)
├── .gitignore
├── download_datasets.py         # Downloads and arranges datasets from Kaggle
├── main.py                      # Training entry point
├── medical_ai.db                # SQLite database (created on first use)
└── requirements.txt
```

`data/` (datasets and uploads) is created automatically on first run, along with `saved_models/`, `logs/`, `assets/` and `medical_ai.db` if they are missing.

## Getting started

### 1. Install

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root:

```env
# Needed only for AI reports
GOOGLE_API_KEY=your_gemini_api_key

# Optional
GEMINI_MODEL=gemini-2.5-flash
LOG_LEVEL=INFO

# Needed only to download datasets (or use ~/.kaggle/kaggle.json)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

The app starts without a Google key; only the "Generate AI report" button is unavailable.

### 3. Download the datasets

```bash
python download_datasets.py
```

This uses `kagglehub` to fetch:

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (you must accept the competition rules on Kaggle with the same account as your API key; the download is several GB)

Datasets are arranged into `data/datasets/<Disease>/{Training,Testing}/<Class>/`. Already-prepared diseases are skipped on re-runs.

### 4. Train the models

```bash
python main.py                   # train all three
python main.py "Pneumonia"       # train only the given disease(s)
```

Each trained model is saved to `saved_models/` (`brain_tumor.keras`, `pneumonia.keras`, `diabetic_retinopathy.keras`) and evaluation results are printed after training. A failure in one disease does not stop the others.

### 5. Run the app

```bash
streamlit run app/Home.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The Home page shows how many of the 3 models are ready.

## Testing

```bash
pytest
```

The suite covers image/PDF loading, dataset loading, risk logic, database behaviour, upload validation, the AI client without a key, and report disclaimers.

## Tech stack

Python · TensorFlow / Keras · ResNet50 · OpenCV · PyMuPDF · scikit-learn · pandas · NumPy · Streamlit · SQLite · Google Gemini (`google-genai`) · kagglehub · pytest

## Limitations

- Models are trained on public datasets and have not been clinically validated.
- Grad-CAM shows where the model looked; it does not prove the prediction is correct.
- AI reports are generated from the predicted class and confidence only, not from the image itself.

## Author

[Aman Satija](https://github.com/amansatija02)