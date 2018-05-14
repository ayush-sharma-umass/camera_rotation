import sys

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt





def load_images():
    dir = '../data/synthetic'
    list_img = []
    filename = 'img_'

    if os.path.exists(dir):
        for i in range(0, 5):
            filepath = dir + '/' + filename + str(i).zfill(3) + '.png'
            img = cv2.imread(filepath)
    #         img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            list_img.append(img)

    imgs = np.asarray(list_img)
    print "Images load. Images_shape(num_images, ht, wt): ", imgs.shape

    return imgs



