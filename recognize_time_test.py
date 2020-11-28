#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Use self-built interface to identify verification codes from the network
Need to configure parameters:
    remote_url = "https://www.xxxxxxx.com/getImg" verification code link address
    rec_times = 1 times of recognition
"""
import datetime
import requests
from io import BytesIO
import time
import json
import os


def recognize_captcha(index, test_path, save_path, image_suffix):
    image_file_name ='captcha.{}'.format(image_suffix)

    with open(test_path, "rb") as f:
        content = f.read()

    # Identification
    s = time.time()
    url = "http://127.0.0.1:6000/b"
    files = {'image_file': (image_file_name, BytesIO(content),'application')}
    r = requests.post(url=url, files=files)
    e = time.time()

    # Test parameters
    result_dict = json.loads(r.text)["value"] # response
    predict_text = result_dict["value"] # Identification result
    whole_time_for_work = int((e-s) * 1000)
    speed_time_by_rec = result_dict["speed_time(ms)"] # Time-consuming model recognition
    request_time_by_rec = whole_time_for_work-speed_time_by_rec # request time
    now_time = datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S') # current time

    # Record log
    log = "{},{},{},{},{},{}\n"\
        .format(index, predict_text, now_time, whole_time_for_work, speed_time_by_rec, request_time_by_rec)
    with open("./test.csv", "a+") as f:
        f.write(log)

    # Output results to the console
    print("Number: {}, Result: {}, Time: {}, Total Time: {}ms, Identification: {}ms, Request: {}ms"
          .format(index, predict_text, now_time, whole_time_for_work, speed_time_by_rec, request_time_by_rec))

    # save document
    # img_name = "{}_{}.{}".format(predict_text, str(time.time()).replace(".", ""), image_suffix)
    # path = os.path.join(save_path, img_name)
    # with open(path, "wb") as f:
    # f.write(content)


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    # Configure related parameters
    test_file = "sample/test/0001_15430304076164024.png" # The path of the picture recognized by the test
    save_path = sample_conf["local_image_dir"] # Saved address
    image_suffix = sample_conf["image_suffix"] # File suffix
    for i in range(20000):
        recognize_captcha(i, test_file, save_path, image_suffix)


if __name__ =='__main__':
    main()