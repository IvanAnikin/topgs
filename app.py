from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from PIL import Image
import numpy as np
import base64
from io import BytesIO

from ML.ml_manager import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():

    try:
        data_url = request.json['image']
        image_data = data_url.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        np_image = np.array(image)

        output_image = detect_people(np_image)

        
        
    except Exception as e:
        return jsonify({'error': str(e)})

    color_info = {
        'coordinates': [(100, 100), (200, 200)],
        'color_name': 'Red',
        'hex_code': '#FF0000'
    }

    print("image received")

    return jsonify(color_info)

    '''
    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    coordinates = [100, 100, 200, 200]
    color_name = "red"
    color_hex = "#FF0000"

    color_data = {
        'coordinates': coordinates,
        'color_name': color_name,
        'color_hex': color_hex
    }
    return jsonify(color_data)
'''
