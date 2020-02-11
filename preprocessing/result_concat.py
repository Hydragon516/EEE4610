import cv2
import glob
import os
import threading

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def make_concat_img(img_path, img, gt, seg):
    seg_ori = seg.copy()

    seg[:,:,0] = 0
    seg[:,:,1] = 0
    seg[:,:,2] = seg[:,:,2] / 2

    seg_sum = cv2.add(img.copy(),seg.copy())
    seg_concat_img = cv2.hconcat([img,seg_sum,seg_ori])

    gt_ori = gt.copy()

    gt[:,:,0] = 0
    gt[:,:,1] = 0
    gt[:,:,2] = gt[:,:,2] / 2

    gt_sum = cv2.add(img.copy(),gt.copy())
    gt_concat_img = cv2.hconcat([img,gt_sum,gt_ori])

    concat_img = cv2.vconcat([gt_concat_img,seg_concat_img])

    concat_img_path = "./concat/" + img_path.split("\\")[-1]
    cv2.imwrite(concat_img_path, concat_img)

gt_list = sorted(glob.glob('./result/GT*.jpg'))
seg_list = sorted(glob.glob('./result/SEG*.jpg'))
img_list = sorted(glob.glob('./result/IMAGE*.jpg'))

for index, img_path in enumerate(img_list):
    gt = cv2.imread(gt_list[index], cv2.IMREAD_COLOR)
    seg = cv2.imread(seg_list[index], cv2.IMREAD_COLOR)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    t_concat = threading.Thread(target=make_concat_img, args=(img_path, img, gt, seg))
    t_concat.start()
