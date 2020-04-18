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

def make_seg_color_edge(img, seg, flag):
    seg_ori = seg.copy()

    seg_ori_white = img.copy()
    seg_ori_white[:,:,0] = seg_ori
    seg_ori_white[:,:,1] = seg_ori
    seg_ori_white[:,:,2] = seg_ori

    seg_ori_r = img.copy()
    if flag == "G":
        seg_ori_r[:,:,0] = 0
        seg_ori_r[:,:,1] = seg_ori
        seg_ori_r[:,:,2] = 0
    elif flag == "B":
        seg_ori_r[:,:,0] = seg_ori
        seg_ori_r[:,:,1] = 0
        seg_ori_r[:,:,2] = 0
    elif flag == "R":
        seg_ori_r[:,:,0] = 0
        seg_ori_r[:,:,1] = 0
        seg_ori_r[:,:,2] = seg_ori

    seg_edge = cv2.Canny(seg, 50, 150)

    seg_color = img.copy()
    if flag == "G":
        seg_color[:,:,0] = 0
        seg_color[:,:,1] = seg_edge
        seg_color[:,:,2] = 0
    elif flag == "B":
        seg_color[:,:,0] = seg_edge
        seg_color[:,:,1] = 0
        seg_color[:,:,2] = 0
    elif flag == "R":
        seg_color[:,:,0] = 0
        seg_color[:,:,1] = 0
        seg_color[:,:,2] = seg_edge

    return seg_color, seg_ori_white, seg_ori_r

def make_concat_img(A_img_path, A_img, A_gt, A_seg, B_seg, B_img):
    

    ret, A_gt = cv2.threshold(A_gt,127,255,cv2.THRESH_BINARY)
    ret, A_seg = cv2.threshold(A_seg,127,255,cv2.THRESH_BINARY)
    ret, B_seg = cv2.threshold(B_seg,127,255,cv2.THRESH_BINARY)

    DSC_A = calc_dsc(A_seg,A_gt)
    DSC_B = calc_dsc(B_seg,A_gt)

    A_seg_color, A_seg_ori_white, A_seg_ori_r = make_seg_color_edge(A_img, A_seg, "G")
    A_seg_sum = cv2.add(A_img.copy(),A_seg_color.copy())
    A_seg_concat_img = cv2.hconcat([A_img,A_seg_sum,A_seg_ori_white])

    B_seg_color, B_seg_ori_white, B_seg_ori_b = make_seg_color_edge(B_img, B_seg, 'B')
    B_seg_sum = cv2.add(B_img.copy(),B_seg_color.copy())
    B_seg_concat_img = cv2.hconcat([B_img,B_seg_sum,B_seg_ori_white])

    gt_color, gt_ori_white, gt_ori_g = make_seg_color_edge(A_img, A_gt, 'R')
    gt_sum = cv2.add(A_img.copy(),gt_color.copy())
    gt_concat_img = cv2.hconcat([A_img,gt_sum,gt_ori_white])
    
    all_ori_rgb = cv2.bitwise_or(cv2.bitwise_or(A_seg_ori_r.copy(), B_seg_ori_b.copy()),gt_ori_g.copy())
    all_seg_sum = cv2.bitwise_or(cv2.bitwise_or(A_seg_color.copy(), B_seg_color.copy()),gt_color.copy())
    all_sum = cv2.add(A_img.copy(),all_seg_sum)

    all_concat_img = cv2.hconcat([A_img,all_sum,all_ori_rgb])

    concat_img = cv2.vconcat([gt_concat_img,A_seg_concat_img,B_seg_concat_img,all_concat_img])
    
    resize_concat_img = cv2.resize(concat_img, (512*3,512*4))

    font = cv2.FONT_ITALIC
    fontScale = 0.8
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(resize_concat_img, 'Ground Truth', (512*2,512*0+30), font, fontScale, color, thickness)
    cv2.putText(resize_concat_img, 'Detect Result', (512*2,512*3+30), font, fontScale, color, thickness)
    cv2.putText(resize_concat_img, 'DSC_A : %.3f' % DSC_A, (512*2,512*1+30), font, fontScale, color, thickness)
    cv2.putText(resize_concat_img, 'DSC_B : %.3f' % DSC_B, (512*2,512*2+30), font, fontScale, color, thickness)

    #cv2.imshow("1",resize_concat_img)
    #cv2.waitKey(0)

    #concat_img_path = "./concat/" + img_path.split("\\")[-1]
    concat_img_path = "./concat/" + "/".join(A_img_path.split("\\")[1:])
    concat_img_path_dsc = concat_img_path.replace(".jpg", "_%.3f_%.3f.jpg" % (DSC_A, DSC_B))
    cv2.imwrite(concat_img_path_dsc, resize_concat_img)
    
def find_same_gt(A_img_path, A_gt, B_gt_list, B_seg_list, B_img_list, B_gt_save):
    file_size = A_img_path.split("\\")[1]
    tmp = np.array(B_gt_save[file_size])
    for index, B_gt in enumerate(tmp[:,0]):
        if np.array_equal(A_gt, B_gt):
            B_seg = cv2.imread(B_seg_list[tmp[index][1]], cv2.IMREAD_GRAYSCALE)
            B_img = cv2.imread(B_img_list[tmp[index][1]], cv2.IMREAD_COLOR)
            del B_gt_save[file_size][index]
            return B_seg, B_img

#gt_list = sorted(glob.glob('./result/GT*.jpg'))
#seg_list = sorted(glob.glob('./result/SEG*.jpg'))
#img_list = sorted(glob.glob('./result/IMAGE*.jpg'))

A_name = "result_Crop_remove_262_12bit"
B_name = "result_Crop_-24_2000"

A_gt_list = sorted(glob.glob('./%s\\*\\GT*.jpg' % A_name))
A_seg_list = sorted(glob.glob('./%s\\*\\SEG*.jpg' % A_name))
A_img_list = sorted(glob.glob('./%s\\*\\IMAGE*.jpg' % A_name))

B_gt_list = sorted(glob.glob('./%s\\*\\GT*.jpg' % B_name))
B_seg_list = sorted(glob.glob('./%s\\*\\SEG*.jpg' % B_name))
B_img_list = sorted(glob.glob('./%s\\*\\IMAGE*.jpg' % B_name))

B_gt_save = {}

for index, img_path in enumerate(A_img_list):
    file_size = img_path.split("\\")[1]
    folder_path = "./concat/" + file_size
    if file_size not in B_gt_save:
        B_gt_save[file_size] = list()
    B_gt = cv2.imread(B_gt_list[index], cv2.IMREAD_GRAYSCALE)
    B_gt_save[file_size].append([B_gt,index])
    make_folder(folder_path)

for index, A_img_path in enumerate(A_img_list):
    A_gt = cv2.imread(A_gt_list[index], cv2.IMREAD_GRAYSCALE)
    A_seg = cv2.imread(A_seg_list[index], cv2.IMREAD_GRAYSCALE)
    A_img = cv2.imread(A_img_path, cv2.IMREAD_COLOR)

    try:
        B_seg, B_img = find_same_gt(A_img_path, A_gt, B_gt_list, B_seg_list, B_img_list, B_gt_save)

        #make_concat_img(A_img_path, A_img, A_gt, A_seg, B_seg, B_img)
        t_concat = threading.Thread(target=make_concat_img, args=(A_img_path, A_img, A_gt, A_seg, B_seg, B_img))
        t_concat.start()
    except:
        print("error : %s" %A_img_path)
