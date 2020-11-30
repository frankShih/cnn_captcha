# -*- coding: UTF-8 -*-
"""
Build flask interface service
Receive files=('image_file': ('captcha.jpg', BytesIO(bytes),'application')} parameter identification verification code
Need to configure parameters:
    image_height = 40
    image_width = 80
    max_captcha = 4
"""
import json
from io import BytesIO
import os
from cnnlib.recognition_object import Recognizer

import time
from flask import Flask, Response, request
from PIL import Image

# Use CPU by default
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Flask object
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Generate recognition object, need to configure parameters

def initialize_rcgz(cfg_path):
    with open(cfg_path, "r") as f:
        sample_conf = json.load(f)
    # Configuration parameters
    image_height = sample_conf["image_height"]
    image_width = sample_conf["image_width"]
    max_captcha = sample_conf["max_captcha"]
    api_image_dir = sample_conf["api_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    image_suffix = sample_conf["image_suffix"] # File suffix
    use_labels_json_file = sample_conf['use_labels_json_file']

    if not os.path.exists(api_image_dir):
        print("[Warning] Cannot find directory {}, will be created soon".format(api_image_dir))
        os.makedirs(api_image_dir)

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    return Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)

# If you need to use multiple models, you can refer to the original example to configure routing and write logic
# Q = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)
rcgz1 = initialize_rcgz("conf/sample1_config.json")
rcgz2 = initialize_rcgz("conf/sample2_config.json")


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] ='*'
    return resp


@app.route('/', methods=['POST'])
def service(req=None):
    if req is None: req=request
    if req.method =='POST' and req.files.get('image_file'):
        file = req.files.get('image_file')
        img = file.read()
        img = BytesIO(img)
        img = Image.open(img, mode="r")
        # username = request.form.get("name")
        print("Receive image size: {}".format(img.size))
        mode = req.args.get('mode', default = None)
        s = time.time()
        if mode=='1':
            value = rcgz1.rec_image(img)
            e = time.time()
            print("Recognition result: {}".format(value))
            img.close()
            result = {
                'timestamp': str(s), # timestamp
                'value': value, # predicted result
                'speed_time(ms)': int((e-s) * 1000) # Identify the time spent
            }
            return Response(json.dumps(result, indent = 4), status=200)
        elif mode=='2':
            value = rcgz2.rec_image(img)
            e = time.time()
            print("Recognition result: {}".format(value))
            img.close()
            result = {
                'timestamp': str(s), # timestamp
                'value': value, # predicted result
                'speed_time(ms)': int((e-s) * 1000) # Identify the time spent
            }
            return Response(json.dumps(result, indent = 4), status=200)
        elif mode=='3':
            return Response('Not implemented yet.', status=204)
        else:
            return Response('Invalid request format', status=404)


    else:
        resp = Response(f'Invalid request', status=404)
        resp.headers['Access-Control-Allow-Origin'] ='*'
        return resp


if __name__ =='__main__':
    app.run(host='0.0.0.0',port=6000,debug=True)
    # app.run(host='0.0.0.0',port=5000,debug=False) # for deployment
