"""
Verify image size and separate test set (5%) and training set (95%)
It is used during initialization. After there is a new picture, you can put the picture in the new directory for use.
"""
import json

from PIL import Image
import random
import os
import shutil


def verify(origin_dir, real_width, real_height, image_suffix):
    """
    Check image size
    :return:
    """
    if not os.path.exists(origin_dir):
        print("[Warning] Cannot find directory {}, will be created soon".format(origin_dir))
        os.makedirs(origin_dir)

    print("Start to verify the original picture collection")
    # Picture real size
    real_size = (real_width, real_height)
    # Picture name list and quantity
    img_list = os.listdir(origin_dir)
    total_count = len(img_list)
    print("Total pictures in the original set: {} sheets".format(total_count))

    # Invalid picture list
    bad_img = []

    # Traverse all pictures for verification
    for index, img_name in enumerate(img_list):
        file_path = os.path.join(origin_dir, img_name)
        # Filter pictures with incorrect suffixes
        if not img_name.endswith(image_suffix):
            bad_img.append((index, img_name, "The file suffix is ​​incorrect"))
            continue

        # Filter image tags that are not standard ***
        # prefix, posfix = img_name.split("_")
        # if prefix == "" or posfix == "":
        # bad_img.append((index, img_name, "Image tag abnormal"))
        # continue

        # Picture cannot be opened normally
        try:
            img = Image.open(file_path)
        except OSError:
            bad_img.append((index, img_name, "The picture cannot be opened normally"))
            continue

        # Picture size is abnormal
        if real_size == img.size:
            print("{} pass".format(index), end='\r')
        else:
            bad_img.append((index, img_name, "The picture size is abnormal: {}".format(img.size)))

    print("====The following {} pictures are abnormal ====".format(len(bad_img)))
    if bad_img:
        for b in bad_img:
            print("[第{}Picture] [{}] [{}]".format(b[0], b[1], b[2]))
    else:
        print("No exception was found ({} pictures in total)".format(len(img_list)))
    print("========end")
    return bad_img


def split(origin_dir, train_dir, test_dir, bad_imgs):
    """
    Separate training set and test set
    :return:
    """
    if not os.path.exists(origin_dir):
        print("[Warning] Cannot find directory {}, will be created soon".format(origin_dir))
        os.makedirs(origin_dir)

    print("Start to separate the original picture sets: test set (5%) and training set (95%)")

    # Picture name list and quantity
    img_list = os.listdir(origin_dir)
    for img in bad_imgs:
        img_list.remove(img)
    total_count = len(img_list)
    print("A total of {} pictures are allocated to the training set and the test set, of which {} pictures are left in the original directory for abnormalities".format(total_count, len(bad_imgs)))

    # Create folder
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Test set
    test_count = int(total_count*0.05)
    test_set = set()
    for i in range(test_count):
        while True:
            file_name = random.choice(img_list)
            if file_name in test_set:
                pass
            else:
                test_set.add(file_name)
                img_list.remove(file_name)
                break

    test_list = list(test_set)
    print("The number of test sets is: {}".format(len(test_list)))
    for file_name in test_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.move(src, dst)

    # Training set
    train_list = img_list
    print("The number of training sets is: {}".format(len(train_list)))
    for file_name in train_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.move(src, dst)

    if os.listdir(origin_dir) == 0:
        print("migration done")


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    # Picture path
    origin_dir = sample_conf["origin_image_dir"]
    new_dir = sample_conf["new_image_dir"]
    train_dir = sample_conf["train_image_dir"]
    test_dir = sample_conf["test_image_dir"]
    # size of the picture
    real_width = sample_conf["image_width"]
    real_height = sample_conf["image_height"]
    # Picture suffix
    image_suffix = sample_conf["image_suffix"]

    for image_dir in [origin_dir, new_dir]:
        print(">>> Start to verify the directory: [{}]".format(image_dir))
        bad_images_info = verify(image_dir, real_width, real_height, image_suffix)
        bad_imgs = []
        for info in bad_images_info:
            bad_imgs.append(info[1])
        split(image_dir, train_dir, test_dir, bad_imgs)


if __name__ =='__main__':
    main()