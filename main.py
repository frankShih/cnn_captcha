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
import gcsfs
import cv2
import numpy as np
from pathlib import Path

# Use CPU by default
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#TODO add hash string back to image file

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
    model_gcp_dir = sample_conf["model_gcp_dir"]
    if  os.path.exists('/tmp'):
        model_save_dir = f'/tmp/{model_save_dir}'

    path = Path(model_save_dir)
    path.mkdir(parents=True, exist_ok=True)
    directory= os.listdir(model_save_dir)
    if len(directory) == 0:
        fs = gcsfs.GCSFileSystem(project=os.environ.get('PROJ_NAME', 'DSU-dev'))
        for filename in fs.ls(model_gcp_dir):
            # print(model_save_dir, model_gcp_dir, filename, filename.split('/')[-1])
            if filename.endswith('/'): continue
            fs.get(filename, '{}/{}'.format(
                model_save_dir, filename.split('/')[-1]))

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
rcgz3 = initialize_rcgz("conf/sample3_config.json")


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] ='*'
    return resp


@app.route('/', methods=['POST'])
def service(req=None):
    if req is None: req=request
    if not req.files.get('image_file'):
        resp = Response(f'Invalid request: image not found', status=404)
        resp.headers['Access-Control-Allow-Origin'] ='*'
        return resp

    s = time.time()
    file = req.files.get('image_file').read()
    img = Image.open(BytesIO(file), mode="r")
    # username = request.form.get("name")
    print("Receive image size: {}".format(img.size))
    mode = req.args.get('mode', default = None)
    if mode=='1':
        value, prob = rcgz1.rec_image(img)
        result = {
            'timestamp': str(s), # timestamp
            'value': value, # predicted result
            'probability': str(prob),
        }
    elif mode=='2':
        value, prob = rcgz2.rec_image(img)
        try:
            new_val=eval(value)
        except Exception as e:
            print(e)
            new_val=''

        result = {
            'timestamp': str(s), # timestamp
            'origin_value': value, # predicted result
            'value': new_val, # predicted result
            'probability': str(prob),
        }
    elif mode=='3':
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2RGBA)
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(open_cv_image, kernel, iterations = 1)
        kernel = np.array([[0, -1, 0], [-1, 11, -1], [0, -1, 0]], np.float32) #锐化
        dilation_flt = cv2.filter2D(erosion, -1, kernel=kernel)
        im_pil = cv2.cvtColor(dilation_flt, cv2.COLOR_BGRA2RGBA)
        im_pil = Image.fromarray(im_pil)
        value, prob = rcgz3.rec_image(im_pil)
        result = {
            'timestamp': str(s), # timestamp
            'value': value, # predicted result
            'probability': str(prob),
        }
    else:
        return Response('Invalid request format', status=404)


    img.close()
    e = time.time()
    res_str = json.dumps(result, indent = 4)
    print(res_str)
    result['speed_time(ms)']=int((e-s) * 1000)
    return Response(res_str, status=200)



if __name__ =='__main__':
    app.run(host='0.0.0.0',port=6000,debug=True)
    # app.run(host='0.0.0.0',port=5000,debug=False) # for deployment
