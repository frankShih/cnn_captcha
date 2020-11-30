# -*- coding: utf-8 -*-
import json

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from PIL import Image
import random
import os
from cnnlib.network import CNN


class TestError(Exception):
    pass


class TestBatch(CNN):
    def __init__(self, img_path, char_set, model_save_dir, total):
        # Model path
        self.model_save_dir = model_save_dir
        # Disorder file order
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        random.seed(time.time())
        random.shuffle(self.img_list)

        # Get basic information about the image width and height and character length
        label, captcha_array = self.gen_captcha_text_image()

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TestError("An error occurred when the picture was converted to a matrix, please check the picture format")

        # Initialize variables
        super(TestBatch, self).__init__(image_height, image_width, len(label), char_set, model_save_dir)
        self.total = total

        # Related information printing
        print("-->Image size: {} X {}".format(image_height, image_width))
        print("-->Verification code length: {}".format(self.max_captcha))
        print("-->Verification codes are {} types {}".format(self.char_set_len, char_set))
        print("-->Use test set as {}".format(img_path))

    def gen_captcha_text_image(self):
        """
        Return an array form of the verification code and the corresponding string label
        :return:tuple (str, numpy.array)
        """
        img_name = random.choice(self.img_list)
        # Tags ***
        # label = img_name.split("_")[0]
        label = img_name.split(".")[0]
        # File
        img_file = os.path.join(self.img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image) # Vectorization

        return label, captcha_array

    def test_batch(self):
        y_predict = self.model()
        total = self.total
        right_word_cnt = 0
        right_char_cnt = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            s = time.time()
            for i in range(total):
                # test_text, test_image = gen_special_num_image(i)
                test_text, test_image = self.gen_captcha_text_image() # random
                test_image = self.convert2gray(test_image)
                test_image = test_image.flatten() / 255

                predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
                text_list = sess.run(predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})
                predict_text = text_list[0].tolist()
                p_text = ""
                for i in range(len(predict_text)):
                    p_char = str(self.char_set[predict_text[i]])
                    if p_char==test_text[i]:
                        p_text +=p_char
                        right_char_cnt+=1
                print("origin: {} predict: {}".format(test_text, p_text))
                if test_text == p_text:
                    right_word_cnt += 1
                else:
                    pass
            e = time.time()
        word_rate = str(right_word_cnt/total * 100) + "%"
        char_rate = right_char_cnt/(total*len(test_text))
        print("Test result: {}/{}".format(right_word_cnt, total))
        print("{} iterations take {} seconds. The character accuracy is {}. The image accuracy is {}".format(total, e-s, word_rate, char_rate))


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    test_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]

    use_labels_json_file = sample_conf['use_labels_json_file']

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    total = 100
    tb = TestBatch(test_image_dir, char_set, model_save_dir, total)
    tb.test_batch()


if __name__ =='__main__':
    main()