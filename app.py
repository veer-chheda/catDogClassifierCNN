from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from flask import request, jsonify

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import PIL
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Load your trained model

model = load_model('catdog.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256,256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
    x=x/255.
    
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = model.predict(x)
    preds = preds[0][0]
    print(preds)
    if preds >= 0.5:
        preds=1
    else:
        preds=0
    if preds==0:
        preds="Cat"
    else:
        preds="Dog"
    
    
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

        # Specify the directory where you want to save the uploaded files
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        
        # Create the directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file to the upload directory
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)