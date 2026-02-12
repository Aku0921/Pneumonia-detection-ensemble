import os
import secrets
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import io

from config import config
from ensemble_predictor import EnsemblePredictor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])

# Initialize database
db = SQLAlchemy(app)

# Initialize predictor (load models on startup)
try:
    predictor = EnsemblePredictor(
        app.config['DENSENET_MODEL'],
        app.config['VGG16_MODEL']
    )
except Exception as e:
    print(f"Warning: Could not load models on startup: {e}")
    predictor = None

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ==================== Database Models ====================

class User(db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password."""
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    """Store prediction history."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    ensemble_prob = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='predictions')


# ==================== Authentication Helpers ====================

def login_required(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    """Check if file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ==================== Routes ====================

@app.route('/')
def index():
    """Home page."""
    if 'user_id' in session:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))
        
        # Create user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome, {user.username}!', 'success')
            return redirect(url_for('upload'))
        
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload and predict X-ray image."""
    if request.method == 'POST':
        if predictor is None:
            flash('Prediction models are not available. Please try again later.', 'danger')
            return redirect(url_for('upload'))
        
        # Check if file is present
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('upload'))
        
        if not allowed_file(file.filename):
            flash('Only PNG, JPG, JPEG, and GIF files are allowed.', 'danger')
            return redirect(url_for('upload'))
        
        try:
            # Read and validate image
            img = Image.open(file.stream).convert('RGB')
            
            # Make prediction
            result = predictor.predict(img)
            
            # Save prediction record
            filename = secure_filename(f"{session['user_id']}_{datetime.utcnow().timestamp()}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            
            prediction = Prediction(
                user_id=session['user_id'],
                image_filename=filename,
                predicted_class=result['class'],
                confidence=result['confidence'],
                ensemble_prob=result['ensemble_prob']
            )
            db.session.add(prediction)
            db.session.commit()
            
            # Store result in session for display
            session['last_result'] = result
            session['last_filename'] = filename
            
            return redirect(url_for('result'))
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('upload'))
    
    return render_template('upload.html')


@app.route('/result')
@login_required
def result():
    """Display prediction result."""
    result = session.get('last_result')
    filename = session.get('last_filename')
    
    if not result:
        flash('No prediction available.', 'warning')
        return redirect(url_for('upload'))
    
    # Determine result message and color
    if result['class'] == 'PNEUMONIA':
        result['message'] = '⚠️ Pneumonia Detected'
        result['color'] = 'danger'
        result['advice'] = 'Please consult a medical professional for further diagnosis and treatment.'
    else:
        result['message'] = '✓ No Pneumonia Detected'
        result['color'] = 'success'
        result['advice'] = 'Your X-ray appears normal. If you have symptoms, please consult a doctor.'
    
    result['image_url'] = url_for('static', filename=f'uploads/{filename}') if filename else None
    
    return render_template('result.html', result=result)


@app.route('/history')
@login_required
def history():
    """View prediction history."""
    user_id = session['user_id']
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    
    return render_template('history.html', predictions=predictions)


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models_loaded': predictor is not None if predictor else False
    })


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


# ==================== CLI Commands ====================

@app.shell_context_processor
def make_shell_context():
    """Add context to Flask shell."""
    return {'db': db, 'User': User, 'Prediction': Prediction}


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True, host='0.0.0.0', port=5000)
