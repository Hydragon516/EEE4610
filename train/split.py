import glob
import os
import numpy as np

def mapping(target, range_min, range_max):
    return (target * (range_max - range_min)) + range_min

def select_list(img_lists, target_num, selected_lung_list, train_flag):
    min_lung = img_lists[0].split("_")[0]
    max_lung = img_lists[-1].split("_")[0]

    target_list = []
    taget_lung = []

    if train_flag:
        for i in range(int(min_lung), int(max_lung) + 1):
            select_lung = "%03d" % i
            pass_flag = True

            for selected_lung in selected_lung_list:
                if select_lung == selected_lung:
                    pass_flag = False

            if pass_flag:
                for img in img_lists:
                    if img.split("_")[0] == select_lung:
                        target_list.append(img)
                        taget_lung.append(select_lung)
    else:
        while len(target_list) < target_num:
            select_lung = "%03d" % int(mapping(np.random.rand(), int(min_lung), int(max_lung)))
            pass_flag = True

            for selected_lung in selected_lung_list:
                if select_lung == selected_lung:
                    pass_flag = False

            if pass_flag:
                for img in img_lists:
                    if img.split("_")[0] == select_lung:
                        target_list.append(img)
                        taget_lung.append(select_lung)

    taget_lung = sorted(list(set(taget_lung)))

    return(taget_lung, target_list)

def split_img_set(img_path, mask_path, val_num, test_num):
    img_lists = np.array(os.walk(img_path).__next__()[2])
    mask_lists = np.array(os.walk(mask_path).__next__()[2])

    val_img_lung, val_img_lists = select_list(img_lists, val_num, [], False)
    test_img_lung, test_img_lists = select_list(img_lists, test_num, val_img_lung, False)
    train_img_lung, train_img_lists = select_list(img_lists, 0, val_img_lung + test_img_lung, True)
    print(len(val_img_lists), len(test_img_lists), len(train_img_lists))

    return val_img_lung, val_img_lists, test_img_lung, test_img_lists, train_img_lung, train_img_lists

if __name__ is "__main__":
    img_path="./Crop_Centered/Crop_image"
    mask_path="./Crop_Centered/Crop_mask"
    val_img_lung, val_img_lists, test_img_lung, test_img_lists, train_img_lung, train_img_lists = split_img_set(img_path, mask_path, 500*4, 1000*4)
