import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from utils.config import MODEL_SAVE_PATH, INPUT_SHAPE

UPLOAD_FOLDER = 'static/uploads/'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model(os.path.join(MODEL_SAVE_PATH, "vgg_finetuned.h5"))

# Mapping of output
CLASS_MAP = {0: "healthy", 1: "malaria"}

def preprocess_image(img_path):
    """
    Preprocess image exactly like during training:
    - Resize
    - RGB
    - Normalize [0,1]
    """
    img = image.load_img(img_path, target_size=INPUT_SHAPE[:2], color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    image_url = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]
        if file.filename == "":
            return "No file selected", 400

        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_url = "/" + filepath

        # Preprocess image
        img_array = preprocess_image(filepath)

        # Predict
        prob = model.predict(img_array)[0][0]  # sigmoid output
        pred_class = 1 if prob > 0.5 else 0

        prediction = CLASS_MAP[pred_class]
        probability = round(prob * 100, 2) if pred_class == 1 else round((1 - prob) * 100, 2)

        # Optional: flag uncertain predictions
        if 0.45 < prob < 0.55:
            prediction += " (uncertain)"

    return render_template("index.html", prediction=prediction, probability=probability, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
