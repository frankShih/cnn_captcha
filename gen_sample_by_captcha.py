# -*- coding: UTF-8 -*-
"""
Use captcha lib to generate verification code (premise: pip install captcha)
"""
from captcha.image import ImageCaptcha
import os
import random
import time
import json


def gen_special_img(text, file_path, width, height):
    # Generate img file
    generator = ImageCaptcha(width=width, height=height) # Specify size
    img = generator.generate_image(text) # Generate image
    img.save(file_path) # save the picture


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # Determine whether the folder exists
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for index, i in enumerate(range(count)):
        text = ""
        for j in range(char_count):
            text += random.choice(characters)

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        print("Generate captcha image => {}".format(index + 1))


def main():
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

    gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height)


if __name__ =='__main__':
    main()