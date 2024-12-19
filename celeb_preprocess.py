import os
import cv2
import numpy as np
import torch
def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = './data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = './data/CelebAMask-HQ/CelebAMask-HQ-mask'
img_num = 7000

make_folder(folder_save)

for k in range(img_num):
    folder_num = k//2000
    target = []
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            # print (label, idx+1)
            im = cv2.imread(filename)
            target.append(im)
    target = np.stack(target, axis=0)
    target[target > 0] = 1
    filename_save = os.path.join(folder_save, str(k) + '.pt')
    print (filename_save)
    torch.save(target, filename_save)