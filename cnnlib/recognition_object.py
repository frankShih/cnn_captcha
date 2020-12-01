# -*- coding: utf-8 -*-
"""
The class for recognizing images, in order to perform multiple recognition quickly, you can call the following methods of this class:
R = Recognizer(image_height, image_width, max_captcha)
for i in range(10):
    r_img = Image.open(str(i) + ".jpg")
    t = R.rec_image(r_img)
Each simple picture can basically reach the recognition speed of milliseconds
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image
from cnnlib.network import CNN
import json
import os
import gcsfs

class Recognizer(CNN):
    def __init__(self, image_height, image_width, max_captcha, char_set, model_save_dir):
        # Initialize variables
        super(Recognizer, self).__init__(image_height, image_width, max_captcha, char_set, model_save_dir)

        # New picture and session
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        # Use the specified graph and session
        with self.g.as_default():
            # Before iterating the loop, write out the calculation expressions of all the tensors used. If written in the loop, memory leaks will occur, slowing down the recognition speed
            # tf initialize placeholder
            self.X = tf.placeholder(tf.float32, [None, self.image_height * self.image_width]) # feature vector
            self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len]) # label
            self.keep_prob = tf.placeholder(tf.float32) # dropout value
            # Load network and model parameters
            self.y_predict = self.model()
            self.predict = tf.argmax(tf.reshape(self.y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            saver = tf.train.Saver()
            with self.sess.as_default() as sess:
                saver.restore(sess, self.model_save_dir)

    # def __del__(self):
    # self.sess.close()
    # print("session close")

    def rec_image(self, img):
        # Read picture
        img_array = np.array(img)
        test_image = self.convert2gray(img_array)
        test_image = test_image.flatten() / 255
        # Use the specified graph and session
        with self.g.as_default():
            with self.sess.as_default() as sess:
                text_list = sess.run(self.predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})

        # Get results
        predict_text = text_list[0].tolist()
        p_text = ""
        for p in predict_text:
            p_text += str(self.char_set[p])

        # Return recognition result
        return p_text


def main():
    with open("conf/sample_config.json", "r", encoding="utf-8") as f:
        sample_conf = json.load(f)
    image_height = sample_conf["image_height"]
    image_width = sample_conf["image_width"]
    max_captcha = sample_conf["max_captcha"]
    char_set = sample_conf["char_set"]
    model_save_dir = sample_conf["model_save_dir"]
    model_gcp_dir = sample_conf["model_gcp_dir"]
    if os.path.exists('/tmp'):
        model_save_dir = f'/tmp/{model_save_dir}'

    fs = gcsfs.GCSFileSystem(project=os.environ.get('PROJ_NAME', 'DSU-dev'))
    for filename in fs.ls(model_gcp_dir):
        if filename.endswith('/'): continue
        fs.get(filename, '{}/{}'.format(
            model_save_dir, filename.split('/')[-1]))

    R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)
    r_img = Image.open("./sample/test/2b3n_6915e26c67a52bc0e4e13d216eb62b37.jpg")
    t = R.rec_image(r_img)
    print(t)


if __name__ =='__main__':
    main()