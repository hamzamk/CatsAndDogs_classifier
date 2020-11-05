from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num of GPUs available: ', len(physical_devices))  

try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  print('Memory growth controlled') 
except:
  # Invalid device or cannot modify virtual devices once initialized.
  print('Device error')

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/'

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(path):
	image = load_img(
    path, color_mode="grayscale", target_size=(224, 224), interpolation="nearest"
)

	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image


def model_predict(img_path, model):
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    img = preprocess_image(img_path)
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = np.round(model_predict(file_path, model))
        if np.round(prediction).squeeze()[1] == 1:
            result = 'Dog'
        if np.round(prediction).squeeze()[1] == 0:
            result = 'Cat'
            
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)


