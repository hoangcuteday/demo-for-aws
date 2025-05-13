import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("mnist_model.keras")

UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prepare_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    image = np.invert(np.array([image]))
    
    image = cv2.resize(image[0], (28, 28))
    
    image = image / 255.0
    image = image.reshape(1, 28, 28)
    
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = 'uploaded_image.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed = prepare_image(filepath)
            pred = model.predict(processed)
            prediction = np.argmax(pred)
            os.remove(filepath)
    else: return render_template('index.html')
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
