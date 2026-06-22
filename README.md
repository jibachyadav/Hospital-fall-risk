# Hospital Patient Fall Risk Detection

A real-time fall risk detection system for hospital patients using computer vision and machine learning.

## Overview

This system uses a webcam feed to detect patient poses in real-time, extracts biomechanical features using MediaPipe, and classifies fall risk into three levels: **safe**, **medium_risk**, and **high_risk**.

## Architecture
## Tech Stack

- **Computer Vision**: MediaPipe (33 body landmarks)
- **ML Model**: XGBoost classifier
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Data Processing**: pandas, numpy, scikit-learn, imbalanced-learn (SMOTE)

## Features Engineered

| Feature | Description |
|---|---|
| spine_angle | Angle at hip between shoulder and knee midpoints |
| lateral_lean | Horizontal distance between shoulder and hip midpoints |
| left_knee_angle | Angle at left knee joint |
| right_knee_angle | Angle at right knee joint |
| hip_visibility | MediaPipe confidence score for hip landmarks |
| knee_visibility | MediaPipe confidence score for knee landmarks |

## Model Performance

- **Accuracy**: 98.89%
- **Classes**: safe, medium_risk, high_risk
- **Training data**: 900 frames (300 per class)
- **Experiment tracking**: MLflow (2 runs logged)

## Project Structure
patient-fall-prediciton/

├── src/

│   ├── ingestion/

│   │   ├── webcam_stream.py

│   │   └── pose_extractor.py

│   └── processing/

│       └── feature_engineer.py

├── training/

│   └── train.py

├── api/

│   └── main.py

├── streamlit_app/

│   └── app.py

├── scripts/

│   ├── record_dataset.py

│   ├── test_pose.py

│   └── test_features.py

├── data/

│   └── raw/

├── models/

│   └── fall_risk_model.json

└── requirements.txt
## Setup

```bash
git clone https://github.com/jibachyadav/Hospital-fall-risk.git
cd Hospital-fall-risk
conda create -n fall-risk python=3.10
conda activate fall-risk
pip install -r requirements.txt
```

## Usage

**1. Record training data:**
```bash
python scripts/record_dataset.py
```

**2. Train model:**
```bash
python src/training/train.py
```

**3. Start API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**4. Start dashboard:**
```bash
streamlit run streamlit_app/app.py
```

**5. View MLflow experiments:**
```bash
mlflow ui --port 5001
```

## Notes

> Training data was recorded with a laptop webcam with limited field of view. For production use, replace with full-body video recordings to improve prediction accuracy.
