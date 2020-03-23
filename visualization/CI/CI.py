import glob
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
       a = 1.0*np.array(data)
       n = len(a)
       m, se = np.mean(a), scipy.stats.sem(a)
       h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
       return m, m-h, m+h

root = glob.glob('./result_CenterCrop_26_2024/*')
print(len(root))

x = []
y = []
err = []
y_0 = []
err_0 = []
dsc_0_num = []

for fold in root:
    size = fold.split("\\")[-1]
    gt_list = sorted(glob.glob(fold+'/GT*.jpg'))
    seg_list = sorted(glob.glob(fold+'/SEG*.jpg'))

    DSC_list = []
    DSC_wo_0_list = []

    for i in range(len(gt_list)):
        
        GT = cv2.imread(gt_list[i], 0) / 255

        GT[GT > 0.5] = 1
        GT[GT <= 0.5] = 0

        AUTO = cv2.imread(seg_list[i], 0) / 255

        AUTO[AUTO > 0.5] = 1
        AUTO[AUTO <= 0.5] = 0

        A_Intersect_G = np.multiply(AUTO, GT)
                
        DSC = (2 * np.sum(A_Intersect_G)) / (np.sum(AUTO) + np.sum(GT))

        if DSC > 0:
            DSC_list.append(DSC)
            DSC_wo_0_list.append(DSC)
        else:
            DSC_list.append(DSC)

    m, nh, ph = mean_confidence_interval(DSC_list, confidence=0.95)
    m_0, nh_0, ph_0 = mean_confidence_interval(DSC_wo_0_list, confidence=0.95)

    x.append(size)
    y.append(m)
    err.append(m-nh)
    
    y_0.append(m_0)
    err_0.append(m_0-nh_0)
    dsc_0_num.append(len(DSC_list) - len(DSC_wo_0_list))

    #print(m, nh, ph)
    #print(m_0, nh_0, ph_0)
    #print("----------------------------")

data = {"mean": y, "err": err,"mean_0": y_0, "err_0": err_0, "dsc 0": dsc_0_num}
df = pd.DataFrame(data, index=x)
df.columns.name = 'crop size'
df.to_excel("table.xlsx")
print(df)

plt.errorbar(x, y, yerr=err, fmt='.k')
plt.show()

plt.errorbar(x, y_0, yerr=err_0, fmt='.k')
plt.show()

print(err)
print(err_0)
