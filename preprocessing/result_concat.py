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

def calc_dsc(seg,gt):
    gt_normal = gt / 255
    seg_normal = seg / 255

    A_Intersect_G = np.multiply(gt_normal, seg_normal)
            
    DSC = (2 * np.sum(A_Intersect_G)) / (np.sum(gt_normal) + np.sum(seg_normal))
    return DSC

def make_concat_img(img_path, img, gt, seg):
    ret, gt = cv2.threshold(gt,127,255,cv2.THRESH_BINARY)
    ret, seg = cv2.threshold(seg,127,255,cv2.THRESH_BINARY)

    DSC = calc_dsc(seg,gt)

    seg_ori = seg.copy()

    seg_ori_white = img.copy()
    seg_ori_white[:,:,0] = seg_ori
    seg_ori_white[:,:,1] = seg_ori
    seg_ori_white[:,:,2] = seg_ori

    seg_ori_r = img.copy()
    seg_ori_r[:,:,0] = 0
    seg_ori_r[:,:,1] = seg_ori
    seg_ori_r[:,:,2] = 0

    seg_edge = cv2.Canny(seg, 50, 150)

    seg_color = img.copy()
    seg_color[:,:,0] = 0
    seg_color[:,:,1] = seg_edge
    seg_color[:,:,2] = 0

    seg_sum = cv2.add(img.copy(),seg_color.copy())
    seg_concat_img = cv2.hconcat([img,seg_sum,seg_ori_white])
    
    gt_ori = gt.copy()

    gt_ori_white = img.copy()
    gt_ori_white[:,:,0] = gt_ori
    gt_ori_white[:,:,1] = gt_ori
    gt_ori_white[:,:,2] = gt_ori

    gt_ori_g = img.copy()
    gt_ori_g[:,:,0] = 0
    gt_ori_g[:,:,1] = 0
    gt_ori_g[:,:,2] = gt_ori

    gt_edge = cv2.Canny(gt, 50, 150)

    gt_color = img.copy()
    gt_color[:,:,0] = 0
    gt_color[:,:,1] = 0
    gt_color[:,:,2] = gt_edge

    gt_sum = cv2.add(img.copy(),gt_color.copy())
    gt_concat_img = cv2.hconcat([img,gt_sum,gt_ori_white])

    all_ori_rg = cv2.add(seg_ori_r.copy(),gt_ori_g.copy())
    all_sum = cv2.add(img.copy(),cv2.add(seg_color.copy(),gt_color.copy()))

    all_concat_img = cv2.hconcat([img,all_sum,all_ori_rg])

    concat_img = cv2.vconcat([gt_concat_img,seg_concat_img,all_concat_img])
    resize_concat_img = cv2.resize(concat_img, (512*3,512*3))

    font = cv2.FONT_ITALIC
    fontScale = 0.8
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(resize_concat_img, 'Ground Truth', (512*2,512*0+30), font, fontScale, color, thickness)
    cv2.putText(resize_concat_img, 'Detect Result', (512*2,512*1+30), font, fontScale, color, thickness)
    cv2.putText(resize_concat_img, 'DSC : %.3f' % DSC, (512*2,512*2+30), font, fontScale, color, thickness)

    #cv2.imshow("1",resize_concat_img)
    #cv2.waitKey(0)

    #concat_img_path = "./concat/" + img_path.split("\\")[-1]
    concat_img_path = "./concat/" + "/".join(img_path.split("\\")[1:])
    concat_img_path_dsc = concat_img_path.replace(".jpg", "_%.3f.jpg" % DSC)
    cv2.imwrite(concat_img_path_dsc, resize_concat_img)

#gt_list = sorted(glob.glob('./result/GT*.jpg'))
#seg_list = sorted(glob.glob('./result/SEG*.jpg'))
#img_list = sorted(glob.glob('./result/IMAGE*.jpg'))

gt_list = sorted(glob.glob('./result\\*\\GT*.jpg'))
seg_list = sorted(glob.glob('./result\\*\\SEG*.jpg'))
img_list = sorted(glob.glob('./result\\*\\IMAGE*.jpg'))

for img_path in img_list:
    folder_path = "./concat/" + img_path.split("\\")[1]
    make_folder(folder_path)

for index, img_path in enumerate(img_list):
    gt = cv2.imread(gt_list[index], cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(seg_list[index], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #make_concat_img(img_path, img, gt, seg)
    t_concat = threading.Thread(target=make_concat_img, args=(img_path, img, gt, seg))
    t_concat.start()
