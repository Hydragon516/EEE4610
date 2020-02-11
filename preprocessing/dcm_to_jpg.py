import matplotlib.pyplot as plt

import pydicom
import pydicom.data

import glob
import os
import shutil

import cv2
import numpy as np
import mritopng

def make_folder(folder_path):
    try:
        if not(os.path.isdir(folder_path)):
            os.makedirs(os.path.join(folder_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def read_attribute(dataset):
    # Normal mode:
    print()
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

    # use .get() if not sure the item exists, and want a default value if missing
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

base = "./NSCLC-Radiomics"
new_path = "./sort"
lung_folder = glob.glob(base + "/*")

for lung_path in lung_folder:
    ## 폴더 탐색해서 dcm 파일경로 수집
    folders = glob.glob(lung_path + "/*")
    for path_name in folders:
        if "CTLUNG" in path_name:
            sub_folders = glob.glob(path_name + "/*")
            for sub_folder in sub_folders:
                if "Segmentation" in sub_folder:
                    Seg_dcm_files = glob.glob(sub_folder + "/*.dcm")
                else:
                    unknown_dcm_files = glob.glob(sub_folder + "/*.dcm")
        #elif "StudyID" in path_name:
        else:
            dcm_files = glob.glob(path_name + "/**/*.dcm")

    # seg dcm 파일 분리 후 저장
    Seg_dataset = pydicom.dcmread(Seg_dcm_files[0])
    print(lung_path.split("\\")[-1], len(Seg_dataset.pixel_array))
    for i in range(len(Seg_dataset.pixel_array)):
        file_path = "/".join(Seg_dcm_files[0].replace(base, new_path).split("\\")[0:-1])
        file_name = file_path + '/%06d' % i + ".png"
        make_folder(file_path)
        if not(os.path.isfile(file_name)):
            cv2.imwrite(file_name, (Seg_dataset.pixel_array)[len(Seg_dataset.pixel_array) - i - 1], [cv2.IMWRITE_PNG_BILEVEL, 1])

    # 단층 사진 dcm -> png
    for index, dcm_file in enumerate(dcm_files):
        dataset = pydicom.dcmread(dcm_file)
        ID_tag = dataset.PatientName.family_name
        file_path = "/".join(dcm_file.replace(base, new_path).split("\\")[0:-1])
        file_name = file_path + "/" + dcm_file.split("\\")[-1].replace("dcm","jpg")
        jpg_files = glob.glob("./jpg/" + ID_tag + "/**/*.jpg")
        jpg_file = jpg_files[index]
        make_folder(file_path)
        if not(os.path.isfile(file_name)):
            shutil.copy(jpg_file, file_name)
