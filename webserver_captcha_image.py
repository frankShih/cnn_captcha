# -*- coding: UTF-8 -*-
"""
    Verification code picture interface, visit `/captcha/1` to get pictures
"""
from captcha.image import ImageCaptcha
import os
import random
from flask import Flask, request, jsonify, Response, make_response
import json
import io


# Flask object
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


with open("conf/captcha_config.json", "r") as f:
    config = json.load(f)
# Configuration parameters
root_dir = config["root_dir"] # Picture storage path
image_suffix = config["image_suffix"] # Image storage suffix
characters = config["characters"] # Character set shown on the picture # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
count = config["count"] # How many samples are generated
char_count = config["char_count"] # The number of characters on the picture

# Set image height and width
width = config["width"]
height = config["height"]


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] ='*'
    return resp


def gen_special_img():
    # Random text
    text = ""
    for j in range(char_count):
        text += random.choice(characters)
    print(text)
    # Generate img file
    generator = ImageCaptcha(width=width, height=height) # Specify size
    img = generator.generate_image(text) # Generate image
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


@app.route('/captcha/', methods=['GET'])
def show_photo():
    if request.method =='GET':
        image_data = gen_special_img()
        response = make_response(image_data)
        response.headers['Content-Type'] ='image/png'
        response.headers['Access-Control-Allow-Origin'] ='*'
        return response
    else:
        pass


if __name__ =='__main__':
    app.run(
        host='0.0.0.0',
        port=6100,
        debug=True
    )