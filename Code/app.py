from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load model and labels
model = load_model("waste_classifier_model.h5")
labels = ['Biodegradable Images', 'Recyclable Images', 'Trash Images']

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------- ROUTES --------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', error="No image part in request")

        img_file = request.files['image']
        if img_file.filename == '':
            return render_template('predict.html', error="No file selected")

        filename = secure_filename(img_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        label = labels[np.argmax(prediction)]

        # Show result
        return render_template('result.html', label=label, image_url=filepath)

    # GET request (when user first visits the page)
    return render_template('predict.html')

# -------------------- MAIN --------------------
if __name__ == '__main__':
    app.run(debug=True)
