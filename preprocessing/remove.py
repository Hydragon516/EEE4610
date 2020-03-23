import numpy as np
import cv2
import glob
import os
import threading
import shutil

def check_min_mask_size(min_mask_size,mask):
    mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    ret, mask_img = cv2.threshold(mask_img,127,255,cv2.THRESH_BINARY)

    # 마스크가 최소 픽셀 갯수를 충족하는지 체크
    pixel_size = int((mask_img/255).sum())
    if pixel_size >= min_mask_size:
        return True
    else:
        return False

def find_slice_num(img_list):
    img_num_list = list()
    lung_num_list = list()
    for img in img_list:
        file_name = img.split("\\")[-1].split(".")[0]
        lung_num = int(file_name.split("_")[0])
        #slice_num = int(file_name.split("_")[1])
        lung_num_list.append(lung_num)

    lung_num_list = np.array(lung_num_list)

    for i in range(lung_num_list.min(), lung_num_list.max() + 1):
        img_num_list.append([i,np.where(lung_num_list == i)[0].shape[0]])

    return img_num_list

def check_remove_list(remove_list, lung_num, slice_num):
    for i in np.where(remove_list[:,0] == lung_num)[0]:
        if remove_list[i][1] == slice_num:
            return False
    return True

def check_remove_all(remove_all, lung_num):
    if np.where(remove_all == lung_num)[0].shape[0] > 0:
        return False
    return True

def copy_file(img,new_img):
    shutil.copy(img, new_img)

def thread_copy(index, img, mask_list, min_mask_size, remove_list, remove_all, img_num_list, mask_num_list):
    img_file_name = img.split("\\")[-1].split(".")[0]
    img_lung_num = int(img_file_name.split("_")[0])
    img_slice_num = int(img_file_name.split("_")[1])
    #reverse_img_slice_num = img_num_list[img_lung_num - 1][1] - img_slice_num
    new_img = img.split("\\")[0].replace("data","remove") + "/" + img_file_name.split("_")[0] + "_" + "%06d" % img_slice_num + ".png"

    mask = mask_list[index]
    mask_file_name = mask.split("\\")[-1].split(".")[0]
    mask_lung_num = int(mask_file_name.split("_")[0])
    mask_slice_num = int(mask_file_name.split("_")[1])
    #reverse_mask_slice_num = mask_num_list[mask_lung_num - 1][1] - mask_slice_num
    new_mask = mask.split("\\")[0].replace("data","remove") + "/" + mask_file_name.split("_")[0] + "_" + "%06d" % mask_slice_num + ".png"

    size_flag = check_min_mask_size(min_mask_size,mask)
    list_flag = check_remove_list(remove_list, img_lung_num, img_slice_num)
    all_flag = check_remove_all(remove_all, img_lung_num)

    if size_flag and list_flag and all_flag:
        #with open("log.txt", "a") as f:
        #    f.write(" ".join([img,new_img,mask,new_mask,"\n"]))
        #print(img,new_img,mask,new_mask)
    #if True:
        copy_file(img,new_img)
        copy_file(mask,new_mask)
    
def remove_trash(img_list):
    trash_list = [14,21,85,95,128,194]
    for img in img_list:
        img_file_name = img.split("\\")[-1].split(".")[0]
        img_lung_num = int(img_file_name.split("_")[0])
        img_slice_num = int(img_file_name.split("_")[1])
        new_img = img.split("\\")[0].replace("data","trash") + "/" + img_file_name.split("_")[0] + "_" + "%06d" % img_slice_num + ".png"

        for trash in trash_list:
            if img_lung_num == trash:
                shutil.move(img, new_img)

def main():
    with open("통합.txt", 'r', encoding="utf-8") as f:
        lines = f.readlines()

    remove_list = list()

    for line in lines:
        data = line.replace("\n","").split("-")
        if len(data) > 1:
            remove_list.append(data)
        else:
            remove_all = data[0].split(":")[-1].replace(" ","").split(",")

    remove_list = np.array(remove_list).astype("int")
    remove_all = np.array(remove_all).astype("int")

    #이미지 경로 설정
    img_list = sorted(glob.glob('./data/image/*.png'))
    mask_list = sorted(glob.glob('./data/mask/*.png'))

    print(len(img_list), len(mask_list))
    print(img_list[0], mask_list[0])

    if len(img_list) == len(mask_list):
        # 마스크 최소 픽셀 갯수
        min_mask_size = 1

        #역순 뒤집기 위해 각 환자별 이미지 갯수 탐색

        img_num_list = find_slice_num(img_list)
        mask_num_list = find_slice_num(mask_list)

        for index, img in enumerate(img_list):
            #thread_copy(index, img, mask_list, min_mask_size, remove_list, remove_all, img_num_list, mask_num_list)
            t_crop = threading.Thread(target=thread_copy, args=(index, img, mask_list, min_mask_size, remove_list, remove_all, img_num_list, mask_num_list))
            t_crop.start()
    else:
        print("데이터의 갯수가 맞지 않습니다.")
        print("잘못된 데이터를 trash 폴더로 옮깁니다.")
        remove_trash(img_list)
        remove_trash(mask_list)
        
if __name__ is "__main__":
    main()
        