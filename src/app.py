import os
import json
import time
import logging
import secrets
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

from .auth import (
    init_db, create_user, verify_user, create_session, 
    get_session_user, delete_session, cleanup_expired_sessions
)
from .predictions import add_prediction, list_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pneumonia Detection API",
    description="Binary classification: Normal vs Pneumonia from chest X-ray images",
    version="1.0.0"
)

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(PROJECT_ROOT, 'static')
uploads_dir = os.path.join(static_dir, 'uploads')
models_dir = os.path.join(PROJECT_ROOT, 'models')

# Ensure static directory exists
os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

# Mount static files FIRST (before adding middleware)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, 'templates'))

# Add session middleware for flash messages
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    expose_headers=["*"],
)

# Model configuration
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.keras'))
EFFICIENTNET_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'efficientnetb0', 'final_model.keras'))
DENSENET_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'densenet121', 'final_model.keras'))
VGG16_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'vgg16', 'final_model.keras'))
MOBILENET_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenetv2', 'final_model.keras'))
GATE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'xray_gate', 'final_model.keras'))

MODEL_VERSION = "2.0.0 (Ensemble)"
IMG_SIZE = (224, 224)
THRESHOLD = 0.6
GATE_THRESHOLD = 0.5
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif"}

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Global model cache
_models_cache = {}
_model_load_times = {}

# Store results temporarily (key: unique_id, value: result_data)
_result_cache = {}


def load_model_cached(model_name="resnet50v2"):
    """Load model once and cache it."""
    if model_name in _models_cache:
        return _models_cache[model_name]
    # Clean up old results (>1 hour old) to prevent memory bloat
    current_time = time.time()
    expired_ids = [k for k, v in _result_cache.items() if current_time - v["timestamp"] > 3600]
    for k in expired_ids:
        del _result_cache[k]
    # Select model path
    model_paths = {
        "efficientnetb0": EFFICIENTNET_MODEL_PATH,
        "densenet121": DENSENET_MODEL_PATH,
        "vgg16": VGG16_MODEL_PATH,
        "mobilenetv2": MOBILENET_MODEL_PATH,
        "xray_gate": GATE_MODEL_PATH,
    }
    
    model_path = model_paths.get(model_name, MODEL_PATH)
    
    try:
        logger.info(f"Loading {model_name} model from {model_path}")
        start = time.time()
        model = tf.keras.models.load_model(model_path)
        load_time = time.time() - start
        _models_cache[model_name] = model
        _model_load_times[model_name] = load_time
        logger.info(f"{model_name} model loaded in {load_time:.2f}s")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def preprocess_image(image_bytes: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """Preprocess image for inference."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(img_size)
        # Keep 0-255 scale to match training input (model has Rescaling(1/255))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, 0)  # add batch dim
        return arr
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


def tta_predict(model, image_array):
    """Return mean prediction across simple TTA variants."""
    img = tf.convert_to_tensor(image_array)
    variants = [
        img,
        tf.image.flip_left_right(img),
        tf.image.rot90(img, k=1),
        tf.image.rot90(img, k=3),
    ]
    batch = tf.concat(variants, axis=0)
    preds = model.predict(batch, verbose=0)
    return float(np.mean(preds))


def predict_ensemble(image_array):
    """Get weighted ensemble predictions from EfficientNetB0 and DenseNet121.

    Uses test-time augmentation (TTA) for more stable predictions.
    """
    try:
        effnet_model = load_model_cached("efficientnetb0")
        densenet_model = load_model_cached("densenet121")
        
        # Get predictions with TTA (already in PNEUMONIA probability format)
        effnet_pred = tta_predict(effnet_model, image_array)
        densenet_pred = tta_predict(densenet_model, image_array)
        
        # Weighted ensemble (90.38% test accuracy with threshold 0.7!)
        # Weights based on individual model AUC scores
        ensemble_pred = (effnet_pred * 0.490 + densenet_pred * 0.510)
        
        logger.info(f"Model outputs (PNEUMONIA probability) - EfficientNet: {effnet_pred:.4f}, DenseNet: {densenet_pred:.4f}")
        logger.info(f"Weighted TTA Ensemble: {ensemble_pred:.4f}")
        
        return {
            "ensemble": ensemble_pred,
            "efficientnetb0": effnet_pred,
            "densenet121": densenet_pred
        }
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise


def is_xray_image(image_array) -> bool:
    """Return True if image looks like an X-ray, using gate model if available."""
    if not os.path.exists(GATE_MODEL_PATH):
        logger.warning("X-ray gate model not found; skipping gate")
        return True
    gate_model = load_model_cached("xray_gate")
    prob = float(gate_model.predict(image_array, verbose=0)[0][0])
    logger.info(f"X-ray gate probability: {prob:.4f}")
    return prob >= GATE_THRESHOLD


def load_metrics_summary():
    """Load per-model metrics and optional analysis."""
    model_names = [
        "efficientnetb0",
        "densenet121",
        "resnet50v2",
        "vgg16",
        "mobilenetv2",
    ]
    metrics = []
    for name in model_names:
        path = os.path.join(models_dir, name, "test_metrics.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                metrics.append(data)
            except Exception as e:
                logger.warning(f"Failed to load metrics for {name}: {e}")

    analysis_path = os.path.join(PROJECT_ROOT, "accuracy_analysis.json")
    analysis = None
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load accuracy_analysis.json: {e}")

    return metrics, analysis


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try loading ensemble models
        load_model_cached("efficientnetb0")
        load_model_cached("densenet121")
        return {
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "models": ["efficientnetb0", "densenet121"],
            "ensemble": "weighted-tta",
            "model_type": "ensemble"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...), session_id: Optional[str] = Cookie(None)):
    """Predict pneumonia probability from uploaded chest X-ray image (protected)."""
    # Check authentication
    user = None
    if session_id:
        user = get_session_user(session_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate file size
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE} bytes")
        
        logger.info(f"Processing image: {file.filename} ({len(contents)} bytes)")
        
        # Preprocess and predict
        x = preprocess_image(contents)
        model = load_model_cached()
        
        start_infer = time.time()
        preds = model.predict(x, verbose=0)
        infer_time = time.time() - start_infer
        
        prob = float(preds[0][0])
        label = "pneumonia" if prob >= 0.5 else "normal"
        confidence = max(prob, 1 - prob)
        
        result = {
            "filename": file.filename,
            "prediction": label,
            "pneumonia_probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "inference_time_ms": round(infer_time * 1000, 2)
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_id: Optional[str] = Cookie(None)):
    """Serve home/login page."""
    user = None
    if session_id:
        user = get_session_user(session_id)
    
    if user:
        return templates.TemplateResponse("upload.html", {"request": request, "user": user})
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    user = verify_user(username, password)
    if user:
        session_id = create_session(user["id"])
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=86400)
        return response
    
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "error": "Invalid username or password"
    })


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serve registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(None)
):
    """Handle registration form submission."""
    # Validate confirm_password
    if not confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Please confirm your password"
        })
    
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Passwords do not match"
        })
    
    user_id = create_user(username, email, password)
    if user_id:
        session_id = create_session(user_id)
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=86400)
        return response
    
    return templates.TemplateResponse("register.html", {
        "request": request,
        "error": "Username or email already exists"
    })


@app.get("/logout")
async def logout(session_id: Optional[str] = Cookie(None)):
    """Handle logout."""
    if session_id:
        delete_session(session_id)
    response = RedirectResponse(url="/login")
    response.delete_cookie(key="session_id")
    return response


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, session_id: Optional[str] = Cookie(None)):
    """Serve upload page (protected)."""
    user = None
    if session_id:
        user = get_session_user(session_id)
    
    if not user:
        return RedirectResponse(url="/login")
    
    response = templates.TemplateResponse("upload.html", {"request": request, "user": user})
    
    # Prevent caching of upload page
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


@app.post("/upload")
async def handle_upload(
    request: Request, 
    file: UploadFile = File(...),
    session_id: Optional[str] = Cookie(None)
):
    """Handle image upload and prediction (protected)."""
    user = None
    if session_id:
        user = get_session_user(session_id)
    
    if not user:
        logger.warning("Upload attempt without valid session")
        return RedirectResponse(url="/login")
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"NEW UPLOAD REQUEST - Processing for user: {user}")
        logger.info(f"{'='*80}")

        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.error("Rejected file: invalid extension")
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "user": user,
                "error": "Only JPG, JPEG, PNG, or GIF images are allowed"
            })

        if not file.content_type or file.content_type not in ALLOWED_MIME_TYPES:
            logger.error("Rejected file: invalid MIME type")
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "user": user,
                "error": "Only JPG, JPEG, PNG, or GIF images are allowed"
            })
        
        # Read file
        contents = await file.read()
        
        # Calculate SHA256 hash of file contents for debugging
        file_hash = hashlib.sha256(contents).hexdigest()[:16]  # First 16 chars of hash
        file_size = len(contents)
        
        logger.info(f"[FILE INFO] Name: {file.filename}")
        logger.info(f"[FILE INFO] Size: {file_size} bytes")
        logger.info(f"[FILE INFO] Hash (SHA256): {file_hash}")
        logger.info(f"[FILE INFO] Content-Type: {file.content_type}")
        
        if len(contents) == 0:
            logger.error("Empty file uploaded")
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "user": user,
                "error": "File is empty"
            })
        
        if len(contents) > MAX_FILE_SIZE:
            logger.error(f"File size {len(contents)} exceeds limit {MAX_FILE_SIZE}")
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "user": user,
                "error": f"File size exceeds {MAX_FILE_SIZE} bytes"
            })

        # Save uploaded image for history viewing
        _, ext = os.path.splitext(file.filename or "")
        ext = ext.lower() if ext else ".jpg"
        upload_name = f"{file_hash}_{int(time.time() * 1000)}{ext}"
        upload_path = os.path.join(uploads_dir, upload_name)
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        # Preprocess and predict (also validates image decoding)
        x = preprocess_image(contents)

        # Gate: only accept chest X-ray images
        if not is_xray_image(x):
            raise HTTPException(status_code=400, detail="Only chest X-ray images are allowed")

        # Gate: only accept chest X-ray images
        if not is_xray_image(x):
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "user": user,
                "error": "Only chest X-ray images are allowed"
            })
        
        start_infer = time.time()
        predictions = predict_ensemble(x)
        infer_time = time.time() - start_infer
        
        # Use ensemble prediction
        ensemble_prob = predictions["ensemble"]
        effnet_prob = predictions["efficientnetb0"]
        densenet_prob = predictions["densenet121"]
        
        # Threshold tuned for better separation
        label = "PNEUMONIA" if ensemble_prob >= THRESHOLD else "NORMAL"
        confidence = max(ensemble_prob, 1 - ensemble_prob)
        
        logger.info(f"=== PREDICTION RESULT ===")
        logger.info(f"EfficientNetB0: {effnet_prob:.4f}")
        logger.info(f"DenseNet121: {densenet_prob:.4f}")
        logger.info(f"Weighted TTA Ensemble: {ensemble_prob:.4f}")
        logger.info(f"Threshold: {THRESHOLD:.2f}")
        logger.info(f"Final Prediction: {label}")
        logger.info(f"Confidence: {confidence:.4f}")
        logger.info(f"======================")
        
        # Prepare result data for template
        result = {
            "color": "danger" if label == "PNEUMONIA" else "success",
            "message": f"Prediction: {label}",
            "class": label,
            "confidence": round(confidence * 100, 2),
            "ensemble_prob": round(ensemble_prob * 100, 2),
            "ensemble_prob_str": f"{round(ensemble_prob * 100, 2)}%",
            "threshold": THRESHOLD,
            "efficientnetb0_prob": float(effnet_prob),
            "densenet121_prob": float(densenet_prob),
            "image_url": None,
            "image_filename": upload_name,
            "models": {
                "EfficientNetB0": round(effnet_prob * 100, 2),
                "DenseNet121": round(densenet_prob * 100, 2),
                "Ensemble": round(ensemble_prob * 100, 2)
            }
        }
        
        logger.info("Rendering result template")
        
        # Generate unique result ID with timestamp to guarantee uniqueness
        result_id = secrets.token_hex(8)
        timestamp_ms = int(time.time() * 1000)
        unique_key = f"{result_id}_{timestamp_ms}"
        
        logger.info(f"\n[RESULT ID GENERATION]")
        logger.info(f"Random token: {result_id}")
        logger.info(f"Timestamp (ms): {timestamp_ms}")
        logger.info(f"Unique Key: {unique_key}")
        
        # Store result with all debugging info
        result["result_id"] = unique_key
        result["file_hash"] = file_hash
        result["file_size"] = file_size
        result["filename"] = file.filename
        result["image_filename"] = upload_name
        _result_cache[unique_key] = {
            "result": result,
            "filename": file.filename,
            "file_hash": file_hash,
            "file_size": file_size,
            "inference_time": round(infer_time * 1000, 2),
            "timestamp": time.time()
        }

        # Store history in database
        add_prediction(
            user_id=user["id"],
            image_filename=upload_name,
            file_hash=file_hash,
            predicted_class=label,
            confidence=round(confidence * 100, 2),
            ensemble_prob=float(ensemble_prob),
            effnet_prob=float(effnet_prob),
            densenet_prob=float(densenet_prob),
        )
        
        logger.info(f"\n[CACHE STORAGE]")
        logger.info(f"Storing result with key: {unique_key}")
        logger.info(f"Cache size: {len(_result_cache)} items")
        logger.info(f"Cached keys: {list(_result_cache.keys())}")
        logger.info(f"\n{'='*80}")
        logger.info(f"UPLOAD COMPLETE - Returning redirect to {unique_key}")
        logger.info(f"{'='*80}\n")
        
        # Check if this is AJAX request
        accept_header = request.headers.get("accept", "")
        is_ajax = (
            "application/json" in accept_header
            or request.headers.get("x-requested-with") == "XMLHttpRequest"
        )
        if is_ajax:
            # AJAX: return JSON with redirect URL
            redirect_url = f"/result/{unique_key}?t={timestamp_ms}"
            logger.info(f"AJAX response: redirecting to {redirect_url}")
            return JSONResponse({
                "redirect": redirect_url
            }, headers={
                "X-Redirect-URL": redirect_url
            })
        else:
            # Traditional: redirect (shouldn't happen with new AJAX form)
            redirect_url = f"/result/{unique_key}"
            logger.info(f"Traditional response: redirecting to {redirect_url}")
            response = RedirectResponse(url=redirect_url, status_code=303)
            return response
        
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user": user,
            "error": f"Prediction failed: {str(e)}"
        })


@app.get("/result/{result_id}", response_class=HTMLResponse)
async def get_result(result_id: str, request: Request, session_id: Optional[str] = Cookie(None)):
    """Retrieve result from cache by unique ID."""
    user = None
    if session_id:
        user = get_session_user(session_id)
    
    if not user:
        return RedirectResponse(url="/login")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RETRIEVING RESULT: {result_id}")
    logger.info(f"{'='*80}")
    
    # Log all cached results
    logger.info(f"[CACHE STATE] Total cached results: {len(_result_cache)}")
    for key, data in _result_cache.items():
        logger.info(f"  - Key: {key}")
        logger.info(f"    File: {data['filename']} ({data['file_size']} bytes, hash: {data.get('file_hash', 'N/A')})")
        logger.info(f"    Age: {time.time() - data['timestamp']:.1f}s")
    
    # Get result from cache
    result_data = _result_cache.get(result_id)
    if not result_data:
        logger.error(f"[ERROR] Result not found in cache!")
        logger.error(f"Requested ID: {result_id}")
        logger.error(f"Available IDs: {list(_result_cache.keys())}")
        logger.info(f"{'='*80}\n")
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user": user,
            "error": "Result not found or expired"
        })
    
    logger.info(f"[FOUND] Result found in cache!")
    logger.info(f"[RESULT DATA]")
    logger.info(f"  File: {result_data['filename']} ({result_data['file_size']} bytes)")
    logger.info(f"  Hash: {result_data.get('file_hash', 'N/A')}")
    logger.info(f"  Inference time: {result_data['inference_time']}ms")
    logger.info(f"  Cached {time.time() - result_data['timestamp']:.1f}s ago")
    logger.info(f"{'='*80}\n")
    
    # Render result with cache-busting headers
    response = templates.TemplateResponse("result.html", {
        "request": request,
        "user": user,
        "result": result_data["result"],
        "filename": result_data["filename"],
        "inference_time": result_data["inference_time"]
    })
    
    # Strong cache prevention
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["ETag"] = result_id  # Unique ETag for each result
    
    return response


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, session_id: Optional[str] = Cookie(None)):
    """Show prediction history for the logged-in user."""
    user = None
    if session_id:
        user = get_session_user(session_id)

    if not user:
        return RedirectResponse(url="/login")

    predictions = list_predictions(user["id"], limit=50)
    return templates.TemplateResponse("history.html", {
        "request": request,
        "user": user,
        "predictions": predictions
    })


@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request, session_id: Optional[str] = Cookie(None)):
    """Show model metrics and charts."""
    user = None
    if session_id:
        user = get_session_user(session_id)

    if not user:
        return RedirectResponse(url="/login")

    metrics, analysis = load_metrics_summary()
    return templates.TemplateResponse("metrics.html", {
        "request": request,
        "user": user,
        "metrics": metrics,
        "analysis": analysis
    })


@app.get("/metrics/roc")
async def metrics_roc():
    """Serve ROC curve chart if available."""
    roc_path = os.path.join(models_dir, "roc.png")
    if not os.path.exists(roc_path):
        raise HTTPException(status_code=404, detail="ROC chart not found")
    return FileResponse(roc_path)


@app.get("/metrics/confusion")
async def metrics_confusion():
    """Serve confusion matrix chart if available."""
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    if not os.path.exists(cm_path):
        raise HTTPException(status_code=404, detail="Confusion matrix not found")
    return FileResponse(cm_path)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """OpenAPI/Swagger docs (auto-generated by FastAPI)."""
    pass  # FastAPI handles this automatically


if __name__ == "__main__":
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Preload ensemble models on startup
    try:
        logger.info("Loading ensemble models...")
        load_model_cached("resnet50v2")
        load_model_cached("densenet121")
        logger.info("✓ Ensemble models preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload ensemble models: {e}")
    
    # Run server
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
