import cv2
import glob
import os
import threading
import numpy as np
import errno

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

img_list = sorted(glob.glob('./remove/image/*.png'))

folder_path = "./8bit/image"
make_folder(folder_path)

for index, img_path in enumerate(img_list):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img_8bit = cv2.convertScaleAbs(img, alpha=(255.0/4095.0))
    img_8bit = img_8bit.astype('uint8')

    convert_path = img_path.replace("remove","8bit")
    cv2.imwrite(convert_path, img_8bit)