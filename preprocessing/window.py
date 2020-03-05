import glob
import numpy as np
import cv2

image_path = "./remove_250/image"
mask_path = "./remove_250/mask"

image_list = glob.glob(image_path + "/*")
mask_list = glob.glob(image_path + "/*")

print("Num of image :", len(image_list))
print("Num of mask :", len(mask_list))

lung_buf = 0
lung_num = []
image_list = []

for mask_dir in mask_list:
    #./remove_250/image\001_000067.jpg
    if lung_buf != int((mask_dir.split("\\")[1]).split("_")[0]):
        lung_num.append(int((mask_dir.split("\\")[1]).split("_")[0]))
        lung_buf = int((mask_dir.split("\\")[1]).split("_")[0])

print("\nFound " + str(len(lung_num)) + " lungs")

print("\nNow search mask data...")

for num in lung_num:
    mask_size_buf = 0
    for mask_dir in mask_list:
        if num == int((mask_dir.split("\\")[1]).split("_")[0]):
            mask = cv2.imread(mask_dir, 0)
            mask_size = np.sum(mask)/255
            if mask_size > mask_size_buf:
                image_buf = mask_dir.split("\\")[1]
                mask_size_buf = mask_size
        
    image_list.append(image_buf)

print("Complete!")

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

def high_and_low(img, high, low):
    img = np.array(img, dtype="uint16")
    img[img > 5] = img[img > 5] + (255 - high)
    img[img < low] = 0
    img[img > 255] = 255
    img = np.array(img, dtype="uint8")

    return img

def onChange(x): 
    pass 

cv2.namedWindow('Setting', cv2.WINDOW_NORMAL) 
cv2.createTrackbar('IMAGE', 'Setting', 0, len(image_list)-1, onChange)
cv2.createTrackbar('LOW', 'Setting', 0, 255, onChange)
cv2.createTrackbar('HIGH','Setting', 0, 255, onChange)
switch = "0: Gray\n1: Jet"
cv2.createTrackbar(switch, 'Setting', 0, 1, onChange)

###
num = 0
img = cv2.imread(image_path + "/" + image_list[num], 1)
img_buf = img.copy()
mask = cv2.imread(mask_path + "/" + image_list[num], 0)
img = make_concat_img(img, mask)
cv2.setTrackbarPos('HIGH','Setting', 255)
low = cv2.getTrackbarPos('LOW', 'Setting') 
high = cv2.getTrackbarPos('HIGH', 'Setting') 
###

while True: 
    img_num = cv2.getTrackbarPos('IMAGE', 'Setting')
    img = img_buf.copy()
    if num != img_num:
        img = cv2.imread(image_path + "/" + image_list[img_num], 1)
        mask = cv2.imread(mask_path + "/" + image_list[img_num], 0)
        img_buf = img.copy()

    img = high_and_low(img, high, low)
    img = make_concat_img(img, mask)
    
    num = img_num

    color_map = cv2.getTrackbarPos(switch, 'Setting') 

    if color_map == 1:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imshow('image', img)

    low = cv2.getTrackbarPos('LOW', 'Setting') 
    high = cv2.getTrackbarPos('HIGH', 'Setting')
    k = cv2.waitKey(1) 

    if k == 27: 
        break 
    
cv2.destroyAllWindows()