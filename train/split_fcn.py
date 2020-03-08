import glob
import os
import numpy as np
import pickle

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
    img_lists = np.array(sorted(os.walk(img_path).__next__()[2]))
    mask_lists = np.array(sorted(os.walk(mask_path).__next__()[2]))

    val_img_lung, val_img_lists = select_list(img_lists, val_num, [], False)
    test_img_lung, test_img_lists = select_list(img_lists, test_num, val_img_lung, False)
    train_img_lung, train_img_lists = select_list(img_lists, 0, val_img_lung + test_img_lung, True)
    print(len(val_img_lists), len(test_img_lists), len(train_img_lists))

    return val_img_lung, val_img_lists, test_img_lung, test_img_lists, train_img_lung, train_img_lists

def split_test_set(img_path, mask_path, test_num):
    img_lists = np.array(sorted(os.walk(img_path).__next__()[2]))
    mask_lists = np.array(sorted(os.walk(mask_path).__next__()[2]))

    test_img_lung, test_img_lists = select_list(img_lists, test_num, [], False)
    train_img_lung, train_img_lists = select_list(img_lists, 0, test_img_lung, True)
    print(len(test_img_lists), len(train_img_lists))
    save_dataset(test_img_lung, test_img_lists, train_img_lung, train_img_lists)

    return test_img_lung, test_img_lists, train_img_lung, train_img_lists

def save_dataset(test_img_lung, test_img_lists, train_img_lung, train_img_lists):
    lung_num = [test_img_lung, test_img_lists, train_img_lung, train_img_lists]
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(lung_num, f)
        print("save dataset")
        
def load_list(img_lists, lung_list):
    load_img_list = list()
    for img in img_lists:
        for lung_num in lung_list:
            if img.split("_")[0] == lung_num:
                load_img_list.append(img)
    return load_img_list

def load_dataset(img_path):
    with open('dataset.pickle', 'rb') as f:
        lung_num = pickle.load(f)
        print("load dataset")

    img_lists = np.array(sorted(os.walk(img_path).__next__()[2]))
    #mask_lists = np.array(sorted(os.walk(mask_path).__next__()[2]))

    test_img_lists = load_list(img_lists, lung_num[0])
    train_img_lists = load_list(img_lists, lung_num[2])
    return lung_num[0], test_img_lists, lung_num[2], train_img_lists

if __name__ is "__main__":
    img_path="./remove_1/image"
    mask_path="./remove_1/mask"
    #val_img_lung, val_img_lists, test_img_lung, test_img_lists, train_img_lung, train_img_lists = split_img_set(img_path, mask_path, 500*4, 1000*4)
    #test_img_lung, test_img_lists, train_img_lung, train_img_lists = split_test_set(img_path, mask_path, 500*4)
    a, b, c ,d = load_dataset(img_path, mask_path)
