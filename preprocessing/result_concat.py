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

    seg_ori_color = img.copy()
    seg_ori_color[:,:,0] = seg_ori
    seg_ori_color[:,:,1] = seg_ori
    seg_ori_color[:,:,2] = seg_ori

    seg_edge = cv2.Canny(seg, 50, 150)

    seg_color = img.copy()
    seg_color[:,:,0] = 0
    seg_color[:,:,1] = 0
    seg_color[:,:,2] = seg_edge

    seg_sum = cv2.add(img.copy(),seg_color.copy())
    seg_concat_img = cv2.hconcat([img,seg_sum,seg_ori_color])
    
    gt_ori = gt.copy()

    gt_ori_color = img.copy()
    gt_ori_color[:,:,0] = gt_ori
    gt_ori_color[:,:,1] = gt_ori
    gt_ori_color[:,:,2] = gt_ori

    gt_edge = cv2.Canny(gt, 50, 150)

    gt_color = img.copy()
    gt_color[:,:,0] = 0
    gt_color[:,:,1] = 0
    gt_color[:,:,2] = gt_edge

    gt_sum = cv2.add(img.copy(),gt_color.copy())
    gt_concat_img = cv2.hconcat([img,gt_sum,gt_ori_color])

    concat_img = cv2.vconcat([gt_concat_img,seg_concat_img])

    concat_img_path = "./concat/" + img_path.split("\\")[-1]
    cv2.imwrite(concat_img_path, concat_img)

gt_list = sorted(glob.glob('./result/GT*.jpg'))
seg_list = sorted(glob.glob('./result/SEG*.jpg'))
img_list = sorted(glob.glob('./result/IMAGE*.jpg'))

for index, img_path in enumerate(img_list):
    gt = cv2.imread(gt_list[index], cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(seg_list[index], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    ret, gt = cv2.threshold(gt,127,255,cv2.THRESH_BINARY)
    ret, seg = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)

    #make_concat_img(img_path, img, gt, seg)
    t_concat = threading.Thread(target=make_concat_img, args=(img_path, img, gt, seg))
    t_concat.start()
