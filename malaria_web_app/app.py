import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# model = load_model("model/ _model.h5") SPECIFY THE MODEL NAME
model = load_model("model/ " <model_name.h5 ")
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224


def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    if pred > 0.5:
        return "Parasitized (Malaria Detected)"
    else:
        return "Uninfected Cell"


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    img_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result = predict_image(path)
            img_path = path

    return render_template("index.html", result=result, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
