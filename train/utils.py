import os
import torch
import cv2
import numpy as np

def save_result(image_size, data_loader, net):
    result_dir = "./result_-334_897/" + str(image_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        if not(os.path.isdir(result_dir)):
            os.makedirs(os.path.join(result_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise


    for num, item in enumerate(data_loader):
        images_tensor = item['image']
        masks_tensor = item['mask']

        inputs = images_tensor.to(device)
        outputs = net(inputs)
        show_out = outputs.detach().cpu().numpy()

        images = images_tensor.numpy()
        masks = masks_tensor.numpy()

        for i in range(len(images)):
            image = images[i].squeeze()
            #image = cv2.convertScaleAbs(image, alpha=(255.0/4095.0))
            
            cv2.imwrite(result_dir + '/' + 'IMAGE' + '_' + str(num) +'_' + str(i+1) + '.jpg', image * 255)
            mask = masks[i].squeeze()
            cv2.imwrite(result_dir + '/' + 'GT' + '_' + str(num) +'_' + str(i+1) + '.jpg', mask * 255)
            seg = show_out[i].squeeze()
            cv2.imwrite(result_dir + '/' + 'SEG' + '_' + str(num) +'_' + str(i+1) + '.jpg', seg * 255)
            i += 1