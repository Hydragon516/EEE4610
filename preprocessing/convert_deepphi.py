import glob
import os
import errno
import shutil
import threading

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def path_convert(convert_name, file_path, patient, size, target_folder):
    convert_file_path = file_path.split("\\")[-1].replace("%.3d_"%patient, "")

    #convert_path = "./%s/%.3d/%s" %(convert_name,patient,size.split("\\")[-1])
    #convert_file = "%s/%s" %(convert_path,convert_file_path)
    convert_path = "./convert/%s/%s/%.3d" %(size.split("\\")[-1],convert_name,patient)
    convert_file = "%s/%s" %(convert_path,convert_file_path)
    make_folder(convert_path)
    shutil.copy(file_path, convert_file)

def thread_fcn(size, target_folder):
    for patient in range(1,423):
        image_list = glob.glob('%s/Image/%.3d_*.png' %(size,patient))
        mask_list = glob.glob('%s/Mask/%.3d_*.png' %(size,patient))

        for image in image_list:
            path_convert("Dataset", image, patient, size, target_folder)
        for mask in mask_list:
            path_convert("Label", mask, patient, size, target_folder)

target_folder = "Crop_remove_262_12bit"

size_list = glob.glob('./%s/*' %target_folder)

for size in size_list:
    #thread_fcn(size, target_folder)
    t_concat = threading.Thread(target=thread_fcn, args=(size, target_folder))
    t_concat.start()
    