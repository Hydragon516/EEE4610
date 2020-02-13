import numpy as np
import cv2
import glob
import os
import threading

def mapping(target, range_min, range_max):
    return (target * (range_max - range_min)) + range_min

def crop_fcn(pixel_size, mask, random_mask_size_range):
    # 마스크 위치 측정
    pixel_location = np.where(mask == 255)
    x_min = pixel_location[1].min()
    x_max = pixel_location[1].max()
    y_min = pixel_location[0].min()
    y_max = pixel_location[0].max()

    x_len = x_max - x_min
    y_len = y_max - y_min

    x_center = np.mean((x_min,x_max))
    y_center = np.mean((y_min,y_max))
    # crop된 이미지에서 마스크 사이즈 결정
    mask_size = mapping(np.random.rand(), random_mask_size_range[0], random_mask_size_range[1])
    # crop될 이미지 사이즈 결정
    crop_size = pixel_size * 100 / mask_size
    crop_len = np.sqrt(crop_size)
    # crop될 이미지에서 마스크의 중심 결정
    x_ratio = [0,0]
    y_ratio = [0,0]

    x_ratio[0] = min((x_center - x_len / 2) / (crop_len - x_len), 1)
    x_ratio[1] = max(-((512 - x_center - x_len / 2) / (crop_len - x_len) - 1), 0)
    y_ratio[0] = min((y_center - y_len / 2) / (crop_len - y_len), 1)
    y_ratio[1] = max(-((512 - y_center - y_len / 2) / (crop_len - y_len) - 1), 0)

    x_random_ratio = mapping(np.random.rand(), x_ratio[1], x_ratio[0])
    y_random_ratio = mapping(np.random.rand(), y_ratio[1], y_ratio[0])

    crop_x_min = x_center - x_len / 2 - x_random_ratio * (crop_len - x_len)
    crop_x_max = x_center + x_len / 2 + (1 - x_random_ratio) * (crop_len - x_len)
    crop_y_min = y_center - y_len / 2 - y_random_ratio * (crop_len - y_len)
    crop_y_max = y_center + y_len / 2 + (1 - y_random_ratio) * (crop_len - y_len)
    crop_range = [crop_y_min,crop_y_max,crop_x_min,crop_x_max]
    return crop_range
    
def write_crop_img(img, mask, pixel_size, random_mask_size_range, target_size, angle_list, random_rotate, file_path):
    crop_range = crop_fcn(pixel_size, mask, random_mask_size_range)
    while (crop_range[3] > 512) or (crop_range[1] > 512) or (crop_range[2] < 0) or (crop_range[0] < 0):
        crop_range = crop_fcn(pixel_size, mask, random_mask_size_range)
        random_mask_size_range[0] = random_mask_size_range[0] * 0.9

    crop_img = cv2.resize(img[int(crop_range[0]):int(crop_range[1]),int(crop_range[2]):int(crop_range[3])], target_size)
    crop_mask = cv2.resize(mask[int(crop_range[0]):int(crop_range[1]),int(crop_range[2]):int(crop_range[3])], target_size)

    if random_rotate:
        random_angle = np.random.choice(angle_list, 1, replace=True)[0]
        matrix = cv2.getRotationMatrix2D((crop_img.shape[0]/2, crop_img.shape[1]/2), random_angle, 1)
        crop_img = cv2.warpAffine(crop_img, matrix, (crop_img.shape[0], crop_img.shape[1]))
        crop_mask = cv2.warpAffine(crop_mask, matrix, (crop_img.shape[0], crop_img.shape[1]))
    cv2.imwrite('./crop_image/' + file_path[0], crop_img)
    cv2.imwrite('./crop_mask/' + file_path[1], crop_mask)

def main():
    #이미지 경로 설정
    img_list = sorted(glob.glob('./data/image2/*.jpg'))
    mask_list = sorted(glob.glob('./data/mask2/*.jpg'))

    print(len(img_list), len(mask_list))
    print(img_list[0], mask_list[0])

    # 마스크 최소 픽셀 갯수
    min_mask_size = 250
    # 마스크가 이미지에서 차지할 퍼센트 비율
    random_mask_size_range = [0.3, 4]
    # 목표 출력 이미지 사이즈
    target_size = (256,256)
    # 같은 이미지 crop 갯수
    crop_count = 4
    # 랜덤 회전 설정
    random_rotate = True
    angle_list = [0,90,180,270]

    file_num_max = len(img_list)

    for file_num in range(file_num_max):
    #for file_num in test_list:
        img = cv2.imread(img_list[file_num], cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_list[file_num], cv2.IMREAD_GRAYSCALE)
        ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        # 마스크가 최소 픽셀 갯수를 충족하는지 체크
        pixel_size = int((mask/255).sum())
        if pixel_size >= min_mask_size:
            for i in range(crop_count):
                file_path = [img_list[file_num].split("\\")[-1].split(".")[0] + '_' + str(i) + '.jpg', mask_list[file_num].split("\\")[-1].split(".")[0] + '_' + str(i) + '.jpg']
                t_crop = threading.Thread(target=write_crop_img, args=(img, mask, pixel_size, random_mask_size_range, target_size, angle_list, random_rotate,file_path))
                t_crop.start()
                #write_crop_img(img, mask, pixel_size, random_mask_size_range, target_size, angle_list, random_rotate,file_path)
        else:
            print(img_list[file_num].split("\\")[-1], " 최소 픽셀 갯수 부족")
if __name__ == "__main__":
    main()