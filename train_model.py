# -*- coding: utf-8 -*-
import json
import logging
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from cnnlib.network import CNN


class TrainError(Exception):
    pass


class TrainModel(CNN):
    def __init__(self, train_img_path, verify_img_path, char_set, model_save_dir, cycle_stop, acc_stop, cycle_save,
                 image_suffix, train_batch_size, test_batch_size, verify=False):
        # Training related parameters
        self.cycle_stop = cycle_stop
        self.acc_stop = acc_stop
        self.cycle_save = cycle_save
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.image_suffix = image_suffix
        char_set = [str(i) for i in char_set]

        # Disrupt file order + verify image format
        self.train_img_path = train_img_path
        self.train_images_list = os.listdir(train_img_path)
        # Check format
        if verify:
            self.confirm_image_suffix()
        # Disorder file order
        random.seed(time.time())
        random.shuffle(self.train_images_list)

        # Validation set file
        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)

        # Get basic information about the image width and height and character length
        label, captcha_array = self.gen_captcha_text_image(train_img_path, self.train_images_list[0])

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TrainError("An error occurred when the picture was converted to a matrix, please check the picture format")

        # Initialize variables
        super(TrainModel, self).__init__(image_height, image_width, len(label), char_set, model_save_dir)

        # Related information printing
        print("-->Image size: {} X {}".format(image_height, image_width))
        print("-->Verification code length: {}".format(self.max_captcha))
        print("-->Verification codes are {} types {}".format(self.char_set_len, char_set))
        print("-->Use the test set as {}".format(train_img_path))
        print("-->Make the verification set {}".format(verify_img_path))

        # test model input and output
        print(">>> Start model test")
        batch_x, batch_y = self.get_batch(0, size=100)
        print(">>> input batch images shape: {}".format(batch_x.shape))
        print(">>> input batch labels shape: {}".format(batch_y.shape))

    @staticmethod
    def gen_captcha_text_image(img_path, img_name):
        """
        Return an array form of the verification code and the corresponding string label
        :return:tuple (str, numpy.array)
        """
        # Tags ***
        # label = img_name.split("_")[0]
        label = img_name.split(".")[0]
        # File
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image) # Vectorization
        return label, captcha_array

    def get_batch(self, n, size=128):
        batch_x = np.zeros([size, self.image_height * self.image_width]) # initialization
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len]) # Initialization

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch-1 <0:
            raise TrainError("The number of images in the training set needs to be greater than the number of images in each batch of training")
        if n> max_batch-1:
            n = n% max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]
        # print("{}:{}".format(s, e))

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array) # Grayscale image
            batch_x[i, :] = image_array.flatten() / 255 # flatten to one-dimensional
            batch_y[i, :] = self.text2vec(label) # Generate oneHot
        return batch_x, batch_y

    def get_verify_batch(self, size=100):
        batch_x = np.zeros([size, self.image_height * self.image_width]) # initialization
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len]) # Initialization

        verify_images = []
        for i in range(size):
            verify_images.append(random.choice(self.verify_images_list))

        for i, img_name in enumerate(verify_images):
            label, image_array = self.gen_captcha_text_image(self.verify_img_path, img_name)
            image_array = self.convert2gray(image_array) # Grayscale image
            batch_x[i, :] = image_array.flatten() / 255 # flatten to one-dimensional
            batch_y[i, :] = self.text2vec(label) # Generate oneHot
        return batch_x, batch_y

    def confirm_image_suffix(self):
        # Verify all file formats before training
        print("Start to verify all picture suffixes")
        for index, img_name in enumerate(self.train_images_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(self.image_suffix):
                raise TrainError('confirm images suffix: you request [.{}] file but get file [{}]'
                                 .format(self.image_suffix, img_name))
        print("All image format verification passed")

    def train_cnn(self):
        y_predict = self.model()
        print(">>> input batch predict shape: {}".format(y_predict.shape))
        print(">>> End model test")
        # Calculate probability loss
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        # Gradient descent
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        # Calculate accuracy
        predict = tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]) # prediction result
        max_idx_p = tf.argmax(predict, 2) # prediction result
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len]), 2) # label
        # Calculate accuracy
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        with tf.name_scope('char_acc'):
            accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.name_scope('image_acc'):
            accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        # Model save object
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # Recovery model
            if os.path.exists(self.model_save_dir):
                try:
                    saver.restore(sess, self.model_save_dir)
                # Determine the error that there is no model file in the model folder
                except ValueError:
                    print("The model folder is empty, a new model will be created")
            else:
                pass
            # Write log
            tf.summary.FileWriter("logs/", sess.graph)

            step = 1
            for i in range(self.cycle_stop):
                batch_x, batch_y = self.get_batch(i, size=self.train_batch_size)
                # Gradient descent training
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                if step% 10 == 0:
                    # Test based on training set
                    # batch_x_test, batch_y_test = self.get_batch(i, size=self.train_batch_size)
                    acc_char = sess.run(accuracy_char_count, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.})
                    print("{}th training >>> ".format(step))
                    print("[Training Set] The character accuracy rate is {:.5f}. The image accuracy rate is {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))

                    # with open("loss_train.csv", "a+") as f:
                    # f.write("{},{},{},{}\n".format(step, acc_char, acc_image, cost_))

                    # Test based on validation set
                    batch_x_verify, batch_y_verify = self.get_verify_batch(size=self.test_batch_size)
                    acc_char = sess.run(accuracy_char_count, feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count, feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    print("[Verification Set] The character accuracy rate is {:.5f}. The image accuracy rate is {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))

                    # with open("loss_test.csv", "a+") as f:
                    # f.write("{}, {},{},{}\n".format(step, acc_char, acc_image, cost_))

                    # Save and stop after the accuracy rate reaches 99%
                    if acc_image> self.acc_stop:
                        saver.save(sess, self.model_save_dir)
                        print("The accuracy of the validation set reaches 99%, and the model is saved successfully")
                        break
                # Save every 500 rounds of training
                if i% self.cycle_save == 0:
                    saver.save(sess, self.model_save_dir)
                    print("Successfully save the model regularly")
                step += 1
            saver.save(sess, self.model_save_dir)

    def recognize_captcha(self):
        label, captcha_array = self.gen_captcha_text_image(self.train_img_path, random.choice(self.train_images_list))

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "origin:" + label, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(captcha_array)
        # Forecast picture
        image = self.convert2gray(captcha_array)
        image = image.flatten() / 255

        y_predict = self.model()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [image], self.keep_prob: 1.})
            predict_text = text_list[0].tolist()

        print("Correct: {} Prediction: {}".format(label, predict_text))
        # Display pictures and forecast results
        p_text = ""
        for p in predict_text:
            p_text += str(self.char_set[p])
        print(p_text)
        plt.text(20, 1,'predict:{}'.format(p_text))
        plt.show()


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    train_image_dir = sample_conf["train_image_dir"]
    verify_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    cycle_stop = sample_conf["cycle_stop"]
    acc_stop = sample_conf["acc_stop"]
    cycle_save = sample_conf["cycle_save"]
    enable_gpu = sample_conf["enable_gpu"]
    image_suffix = sample_conf['image_suffix']
    use_labels_json_file = sample_conf['use_labels_json_file']
    train_batch_size = sample_conf['train_batch_size']
    test_batch_size = sample_conf['test_batch_size']

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    if not enable_gpu:
        # Set the following environment variables to enable CPU recognition
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tm = TrainModel(train_image_dir, verify_image_dir, char_set, model_save_dir, cycle_stop, acc_stop, cycle_save,
                    image_suffix, train_batch_size, test_batch_size, verify=False)
    tm.train_cnn() # Start training model
    # tm.recognize_captcha() # Recognition picture example


if __name__ =='__main__':
    main()