import glob
import os
import errno
import shutil
import threading
import pickle

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def path_convert(convert_name, file_path, patient, size, target_folder, test_num, train_num):
    convert_file_path = file_path.split("\\")[-1].replace("%.3d_"%patient, "")

    #convert -> size -> Dataset, Label -> patient
    #convert_path = "./convert/%s/%s/%.3d" %(size.split("\\")[-1],convert_name,patient)
    #convert_file = "%s/%s" %(convert_path,convert_file_path)

    #Dataset, Label -> size -> patient
    #convert_path = "./%s/%.3d/%s" %(convert_name,patient,size.split("\\")[-1])
    #convert_file = "%s/%s" %(convert_path,convert_file_path)

    #test, train -> size -> Dataset, Label
    if "%.3d" %patient in test_num:
        convert_path = "./test/%s/%s" %(size.split("\\")[-1],convert_name)
    else:
        convert_path = "./train/%s/%s" %(size.split("\\")[-1],convert_name)

    convert_file = "%s/%.3d_%s" %(convert_path,patient,convert_file_path)
    make_folder(convert_path)
    shutil.copy(file_path, convert_file)

def thread_fcn(size, target_folder, test_num, train_num):
    for patient in range(1,423):
        image_list = glob.glob('%s/Image/%.3d_*.png' %(size,patient))
        mask_list = glob.glob('%s/Mask/%.3d_*.png' %(size,patient))

        for image in image_list:
            path_convert("Dataset", image, patient, size, target_folder, test_num, train_num)
        for mask in mask_list:
            path_convert("Label", mask, patient, size, target_folder, test_num, train_num)

def load_dataset():
    with open('dataset.pickle', 'rb') as f:
        test_num, _, train_num, _ = pickle.load(f)
        print("load dataset")
    
    return test_num, train_num

target_folder = "Crop_remove_262_12bit"

size_list = glob.glob('./%s/*' %target_folder)

test_num, train_num = load_dataset()

for size in size_list:
    if size.split("\\")[-1] == "144":
        thread_fcn(size, target_folder, test_num, train_num)
    #t_concat = threading.Thread(target=thread_fcn, args=(size, target_folder, test_num, train_num))
    #t_concat.start()
    