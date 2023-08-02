from flask import Flask, request, render_template, jsonify
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64

app = Flask(__name__, static_url_path='/static')
@app.route("/")
def index():
    return render_template("index.html")
MODEL = tf.keras.models.load_model("./savedmodels/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Encode the uploaded image as base64
    image_base64 = base64.b64encode(file.read()).decode("utf-8")

    return jsonify({'class': predicted_class, 'confidence': confidence, 'image': image_base64})
if __name__ == "__main__":
    app.run()

