import glob
import numpy as np
import cv2
import os

image_path = "./remove/image"
mask_path = "./remove/mask"

image_list = glob.glob(image_path + "/*")
mask_list = glob.glob(mask_path + "/*")

print("Num of image :", len(image_list))
print("Num of mask :", len(mask_list))

lung_buf = 0
lung_list = []
image_list = []

for mask_dir in mask_list: #./remove_250/image\001_000067.jpg
    image_list.append(mask_dir.split("\\")[1])
    if lung_buf != int((mask_dir.split("\\")[1]).split("_")[0]):
        lung_list.append(int((mask_dir.split("\\")[1]).split("_")[0]))
        lung_buf = int((mask_dir.split("\\")[1]).split("_")[0])

print("\nFound " + str(len(lung_list)) + " lungs")
print("Found " + str(len(image_list)) + " images")

def make_concat_img(img, gt):
    ret, gt = cv2.threshold(gt,127,255,cv2.THRESH_BINARY)

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

    return gt_sum

def window_set(img, WL, WW):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    img = img - WL
    img = img * 4096 / WW
    img = img + 4096 / 2

    img[img < 0] = 0
    img[img > 4095] = 4095
    img = img.astype('uint16')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def save_image(current_list, WL, WW):
    result_dir = "./save/" + str(current_list[0].split("_")[0])

    try:
        if not(os.path.isdir(result_dir)):
            os.makedirs(os.path.join(result_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    for img_name in current_list:
        img = cv2.imread(image_path + "/" + img_name, cv2.IMREAD_ANYDEPTH)
        img = window_set(img, HU_low, HU_high)
        cv2.imwrite(result_dir + "/" + img_name, img)
        print("save " + img_name)

def draw_bar(HU_bar, WL, WW):
    HU_bar = cv2.rectangle(HU_bar, (int((WL - WW / 2) / 4096 * 512), 0), \
        (int((WL + WW / 2) / 4096 * 512), 50), (0, 0, 255), 2)
    
    return HU_bar

def onChange(x): 
    pass 

def load_list(image_list, lung_num):
    load = []
    for fname in image_list:
        if int(fname.split("_")[0]) == lung_num:
            load.append(fname)

    return load


cv2.namedWindow('Setting', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Setting', 600, 200)

current_list = load_list(image_list, 1)

cv2.createTrackbar('LUNG', 'Setting', 0, len(lung_list)-1, onChange)
cv2.createTrackbar('IMAGE', 'Setting', 0, len(current_list)-1, onChange)

cv2.createTrackbar('WL', 'Setting', 0, 4095, onChange)
cv2.setTrackbarPos('WL','Setting', 2048)
cv2.createTrackbar('WW','Setting', 0, 4095, onChange)
cv2.setTrackbarPos('WW','Setting', 4095)

switch = "Gray-Jet"
cv2.createTrackbar(switch, 'Setting', 0, 1, onChange)

###
lung_num_buf = 0
img_num_buf = 0
WL_buf = -1
WW_buf = -1
img = cv2.imread(image_path + "/" + image_list[0], cv2.IMREAD_ANYDEPTH)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_buf = img.copy()
show_buf = img.copy()
mask = cv2.imread(mask_path + "/" + image_list[0], 0)
img = make_concat_img(img, mask)

histogram = np.zeros((50, 512, 3), np.uint8)
for i in range(512):
    histogram[:, i] = 255 / 512 * i
###

while True:
    setting_img = np.zeros((50, 512, 3), np.uint8)
    for i in range(512):
        histogram[:, i] = 255 / 512 * i

    lung_num = cv2.getTrackbarPos('LUNG', 'Setting')
    img_num = cv2.getTrackbarPos('IMAGE', 'Setting')
    WL = cv2.getTrackbarPos('WL', 'Setting') 
    WW = cv2.getTrackbarPos('WW', 'Setting')

    img = img_buf.copy()

    if lung_num_buf != lung_num:
        cv2.setTrackbarPos('IMAGE','Setting', 0)
        img_num = 0
        img_num_buf = 0
        current_list = load_list(image_list, lung_list[lung_num])
        cv2.createTrackbar('IMAGE', 'Setting', 0, len(current_list)-1, onChange)

        img = cv2.imread(image_path + "/" + current_list[img_num], cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = cv2.imread(mask_path + "/" + current_list[img_num], 0)
        
        img_buf = img.copy()
        show_buf = window_set(img_buf, WL, WW)
        lung_num_buf = lung_num
    
    else:
        pass

    if img_num_buf != img_num:
        current_list = load_list(image_list, lung_list[lung_num])
        img = cv2.imread(image_path + "/" + current_list[img_num], cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = cv2.imread(mask_path + "/" + current_list[img_num], 0)

        img_buf = img.copy()
        show_buf = window_set(img_buf, WL, WW)
        img_num_buf = img_num
    
    else:
        pass

    show_img = show_buf.copy()

    if WL_buf != WL or WW_buf != WW:
        if (WL - WW / 2 > 0) and WL + WW / 2 < 4096:
            show_img = img.copy()
            show_img = window_set(show_img, WL, WW)
            show_buf = show_img.copy()

            WL_buf = WL
            WW_buf = WW

        else:
            cv2.setTrackbarPos('WL','Setting', WL_buf)
            cv2.setTrackbarPos('WW','Setting', WW_buf)

    show_img = cv2.convertScaleAbs(show_img, alpha=(255.0/4095.0))
    show_img = show_img.astype('uint8')
    
    show_img = make_concat_img(show_img, mask)
    
    lung_num_buf = lung_num

    color_map = cv2.getTrackbarPos(switch, 'Setting') 

    if color_map == 1:
        show_img = cv2.applyColorMap(show_img, cv2.COLORMAP_JET)

    cv2.putText(setting_img, "Lung : " + str(lung_list[lung_num]), \
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(setting_img, "IMAGE : " + str(current_list[img_num]), \
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    
    HU_bar = draw_bar(histogram, WL, WW)
    show_img = cv2.vconcat([show_img, setting_img])
    show_img = cv2.vconcat([show_img, HU_bar])
    cv2.imshow('image', show_img)

    if cv2.waitKey(1) == 27: 
        break 
    
cv2.destroyAllWindows()