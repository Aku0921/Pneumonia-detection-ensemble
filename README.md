# Pneumonia_Detection-Ensemble

Pneumonia detection website using ensemble models (FastAPI + TensorFlow/Keras).

## Quick Start

1. Extract the Kaggle dataset into `data/` with this layout:

```
data/chest_xray/train/NORMAL
data/chest_xray/train/PNEUMONIA
data/chest_xray/val/...
data/chest_xray/test/...
```

2. Activate your venv and install requirements:

```powershell
\.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run training:

```powershell
python -m src.train --backbone efficientnetb0
```

4. Run inference on a single image:

```powershell
python -m src.infer .\models\final_model.keras data\chest_xray\test\NORMAL\IM-0001-0001.jpeg
```

## Web App

Start the server:

```powershell
\.\run_server.ps1
```

Open:
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### New Pages
- `/history` — past uploads + predictions
- `/metrics` — model metrics and charts

## Evaluation

Generate ROC and confusion matrix:

```powershell
python -m src.evaluate --model .\models\final_model.keras --out_dir models
```

Charts will be available at:
- `/metrics/roc`
- `/metrics/confusion`

## Notes

- `data/`, `models/`, and `static/uploads/` are ignored by Git.
- Uploads are stored in `static/uploads/` for history view.

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── app.py            # FastAPI backend
│   ├── data.py           # Dataset loader
│   ├── train.py          # Training
│   ├── evaluate.py       # ROC/confusion matrix
│   ├── infer.py          # Single-image inference CLI
│   ├── models.py         # Backbone factory
│   ├── ensemble.py       # Ensemble selection
│   └── predictions.py    # Prediction history storage
├── templates/            # Jinja2 templates
├── static/               # CSS/JS/uploads
├── models/               # Trained models (ignored in git)
├── data/                 # Dataset (ignored in git)
└── README.md
```
│   └── test_api.py       # API test script (NEW)
├── Dockerfile            # Container image (NEW)
├── docker-compose.yml    # Local docker setup (NEW)
├── run_server.ps1        # Quick start script (NEW)
├── requirements.txt
└── README.md
```

## 📊 Model Performance

- **Architecture**: Transfer learning with EfficientNetB0
- **Training**: Staged fine-tuning (head → full model)
- **Test ROC-AUC**: ~0.72
- **Inference Time**: ~40-50ms on CPU

## ⚠️ Disclaimers

⚠️ **This model is for research/educational purposes only.** It should not be used for clinical diagnosis without review by a qualified radiologist. Always consult medical professionals for actual patient care decisions.

## 🔄 Next Steps

### Short-term
- [ ] Run multi-backbone experiments (ResNet, MobileNetV2, VGG16)
- [ ] Create ensemble from best 2 models
- [ ] Add EDA and results notebooks

### Medium-term
- [ ] Improve performance (more epochs, class weighting, better augmentation)
- [ ] Add model versioning and A/B testing
- [ ] Deploy to cloud (Render, Cloud Run, or Azure)

### Long-term
- [ ] Add more model architectures and hyperparameter tuning
- [ ] Implement ONNX export for edge deployment
- [ ] Add user authentication and request logging
- [ ] Monitor model drift and retrain regularly
>>>>>>> 23760c9 (Initial commit)
