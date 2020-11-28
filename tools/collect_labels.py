#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Count the labels of the samples and write them into the labels.json file
"""
import os
import json


image_dir = "../sample/origin"
image_list = os.listdir(image_dir)

labels = set()
for img in image_list:
     split_result = img.split("_")
     if len(split_result) == 2:
         label, name = split_result
         if label:
             for word in label:
                 labels.add(word)
     else:
         pass

print("Total tags{} species".format(len(labels)))

with open("./labels.json", "w") as f:
     f.write(json.dumps("".join(list(labels)), ensure_ascii=False))

print("The label list is written to the file labels.json successfully")