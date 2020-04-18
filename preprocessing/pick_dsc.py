import glob
import os
import errno
import shutil

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

folder_name = "concat_compare_-24_2000"
diff = 0.5

image_list = sorted(glob.glob('./%s\\*\\IMAGE*.jpg' % folder_name))

for index, img_path in enumerate(image_list):
    file_size = img_path.split("\\")[1]
    folder_path = "./high/" + file_size
    make_folder(folder_path)
    folder_path = "./low/" + file_size
    make_folder(folder_path)

for image_path in image_list:

    file_name = image_path.split("\\")[-1]
    DSC_A = float(file_name.split("_")[-2])
    DSC_B = float(file_name.split("_")[-1].replace(".jpg",""))

    if DSC_A - DSC_B > diff:
        new_path = image_path.replace(folder_name, "high")
        shutil.copy(image_path, new_path)
    elif DSC_B - DSC_A > diff:
        new_path = image_path.replace(folder_name, "low")
        shutil.copy(image_path, new_path)