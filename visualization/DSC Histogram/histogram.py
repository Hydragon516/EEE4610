import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

def save_dsc_fig(gt_list, seg_list, folder):
    DSC_list = []
    GT_sum = []

    for i in range(len(gt_list)):
        
        GT = cv2.imread(gt_list[i], 0) / 255

        GT[GT > 0.5] = 1
        GT[GT <= 0.5] = 0

        AUTO = cv2.imread(seg_list[i], 0) / 255

        AUTO[AUTO > 0.5] = 1
        AUTO[AUTO <= 0.5] = 0

        A_Intersect_G = np.multiply(AUTO, GT)
                
        DSC = (2 * np.sum(A_Intersect_G)) / (np.sum(AUTO) + np.sum(GT))

        DSC_list.append(DSC)
        GT_sum.append(np.sum(GT))

    print(len(DSC_list))

    all_dsc = "./dsc/" + folder + "_all.png"
    none0_dsc = "./dsc/" + folder + "_none0.png"
    '''
    plt.clf()

    plt.hist(DSC_list, range=(0.1, 1), bins=100)

    plt.xlabel('DSC Values')
    plt.ylabel('Nodule Amount')
    plt.title('Histogram of DSC')

    plt.savefig(none0_dsc, dpi=300)
    '''
    #plt.clf()
    bins = np.arange(0,1+0.05,0.05)
    hist, bins = np.histogram(DSC_list, bins)
    plt.plot(bins[1:], hist, label=folder)

    plt.xlabel('DSC Values')
    plt.ylabel('Nodule Amount')
    plt.title('Histogram of DSC')

    #plt.savefig(all_dsc, dpi=300)
    
    #plt.show()

folder_list = os.walk("./result").__next__()[1]
for folder in folder_list:
    gt_list = sorted(glob.glob('./result\\' + folder + '\\GT*.jpg'))
    seg_list = sorted(glob.glob('./result\\' + folder + '\\SEG*.jpg'))
    print(len(gt_list), len(seg_list))

    save_dsc_fig(gt_list, seg_list, folder)
plt.legend()
plt.show()