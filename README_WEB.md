# 🫁 Pneumonia Detection Web Application

A modern web application for detecting pneumonia from chest X-ray images using a trained ensemble of deep learning models (DenseNet121 + VGG16).

## Features

✅ **User Authentication**
- Secure user registration and login
- Session management
- Password hashing

✅ **Image Upload & Analysis**
- Drag-and-drop image upload
- Real-time image preview
- X-ray analysis using ensemble model

✅ **Prediction Results**
- Pneumonia/Normal classification
- Confidence scores
- Individual model predictions (DenseNet121 + VGG16)
- Visual ROC curves (optional)

✅ **Prediction History**
- Track all user predictions
- View historical X-ray images
- Timestamped results

✅ **Modern UI**
- Bootstrap 5 responsive design
- Clean, intuitive interface
- Mobile-friendly layout

## Project Structure

```
pneumonia-detection/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── ensemble_predictor.py           # Ensemble model wrapper
├── requirements-web.txt            # Python dependencies
├── pneumonia_app.db               # SQLite database (created on first run)
├── templates/                      # HTML templates
│   ├── base.html                  # Base template with navbar
│   ├── login.html                 # Login page
│   ├── register.html              # Registration page
│   ├── upload.html                # Upload and analysis page
│   ├── result.html                # Results display page
│   ├── history.html               # Prediction history page
│   ├── 404.html                   # 404 error page
│   └── 500.html                   # 500 error page
├── static/
│   ├── css/
│   │   └── style.css              # Custom CSS styling
│   ├── js/
│   │   └── script.js              # JavaScript utilities
│   └── uploads/                   # User uploaded images (created on first run)
├── models/
│   ├── densenet121/               # DenseNet121 model
│   │   └── final_model.keras
│   └── vgg16/                     # VGG16 model
│       └── final_model.keras
└── src/                            # Original training code (untouched)
    ├── train.py
    ├── models.py
    ├── data.py
    ├── evaluate.py
    ├── ensemble.py
    └── infer.py
```

## Installation

### 1. Install Web Dependencies

```bash
pip install -r requirements-web.txt
```

### 2. Ensure Models Are Present

The app expects trained models at:
- `models/densenet121/final_model.keras`
- `models/vgg16/final_model.keras`

If models are missing, the app will display an error message.

## Running the Application

### Development Mode

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Production Mode

Set `FLASK_ENV=production` and use a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Usage

### 1. Create Account
- Go to "Register" and create a new account
- Provide username, email, and password

### 2. Login
- Log in with your credentials

### 3. Upload X-Ray Image
- Click "Upload" in the navigation menu
- Drag and drop or browse for a chest X-ray image
- Supported formats: PNG, JPG, JPEG, GIF
- Maximum file size: 16MB

### 4. View Results
- Results display automatically after analysis
- See prediction confidence and individual model outputs
- Medical advice based on results

### 5. Check History
- View all your previous predictions
- Access uploaded images
- Track prediction trends

## Technical Details

### Ensemble Model

The application uses a **2-model averaging ensemble**:
- **DenseNet121**: Trained on chest X-ray dataset
- **VGG16**: Trained on same dataset
- **Strategy**: Average probability predictions from both models
- **Classification**: Threshold at 0.5 probability

### Database

- **Type**: SQLite (file-based)
- **Tables**: 
  - `user` - User accounts and authentication
  - `prediction` - Prediction history and results

### Image Processing

- **Input Size**: 224×224×3 (RGB)
- **Preprocessing**: Rescaling to [0, 1]
- **Augmentation**: Applied during training (not in inference)

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home redirect |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/logout` | GET | User logout |
| `/upload` | GET, POST | Image upload and prediction |
| `/result` | GET | View last prediction result |
| `/history` | GET | View prediction history |
| `/api/health` | GET | Health check endpoint |

## Configuration

Edit `config.py` to customize:

```python
SECRET_KEY              # Flask secret key
SQLALCHEMY_DATABASE_URI # Database location
UPLOAD_FOLDER          # Upload directory
ALLOWED_EXTENSIONS     # Allowed file types
MAX_CONTENT_LENGTH     # Max upload size
MODELS_DIR             # Models directory
```

## Troubleshooting

### Models Not Loading
- Check that model files exist at configured paths
- Verify `safe_mode=False` is set in ensemble predictor
- Check for GPU/CUDA compatibility issues

### Database Issues
- Delete `pneumonia_app.db` to reset database
- Check file permissions in project directory

### Upload Errors
- Ensure uploads directory has write permissions
- Check file size (max 16MB)
- Verify image format is PNG, JPG, JPEG, or GIF

### Memory Issues
- Reduce batch size in ensemble predictor
- Use GPU for faster inference
- Increase available system memory

## Development

### Adding New Features

1. **Add new route** in `app.py`
2. **Create new template** in `templates/`
3. **Add styling** to `static/css/style.css`
4. **Add database models** if needed

### Testing

```bash
# Create test user
python
>>> from app import db, User
>>> user = User(username='test', email='test@example.com')
>>> user.set_password('password')
>>> db.session.add(user)
>>> db.session.commit()
```

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. The predictions should NOT be used for medical diagnosis or treatment decisions. Always consult a qualified medical professional for actual medical diagnosis and treatment.

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Flask documentation: https://flask.palletsprojects.com/
3. Check TensorFlow docs: https://www.tensorflow.org/

---

**Created**: January 2026  
**Framework**: Flask 3.0  
**Models**: DenseNet121 + VGG16 Ensemble  
**Database**: SQLite
