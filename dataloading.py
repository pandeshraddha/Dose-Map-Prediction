import numpy as np
from glob import glob
import os
import SimpleITK as sitk
import random
import imageio
import tensorflow as tf
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
img_res=(256, 256)
in_channels = 6
out_channels = 1

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
    def load_batch(self, batch_size=1, is_testing=False):
        #data_type = "train" if not is_testing else "val"
        #path = glob('../input/%s/%s/%s/*' % (self.dataset_name, self.dataset_name, data_type))
        #path = glob('../%s/*' % (data_type))
        path = 'F:\\Desktop\\Documeents\\GAN\\download_data\\BRATS2015_Training\\BRATS2015_Training\\my_bcmb\\global_scalingrtgt20\\test\\T1\\'
        pa = os.listdir(path)
        pa.sort(key=str.lower)
        n_batches = int(len(pa) / batch_size)
        patha = 'F:\\Desktop\\Documeents\\GAN\\download_data\\BRATS2015_Training\\BRATS2015_Training\\my_bcmb\\global_scalingrtgt20\\test\\'
        for j in range(n_batches - 1):
            batch = pa[j * batch_size:(j + 1) * batch_size]
            for img in batch:
                p = os.listdir(patha)
                p.sort(key=str.lower)
                arr = []
                for i in range(len(p)):
                    if ((i == 0) or (i == 1) or (i == 7) or (i == 8) or (i == 9) or (i == 10)):
                        img_A = sitk.ReadImage(patha +'/'+ p[i] + '/' + img)
                        arr.append(sitk.GetArrayFromImage(img_A))
                    elif (i == 2):
                        img_B = sitk.ReadImage(patha +'/'+ p[i] + '/' + img)
                        img_B = sitk.GetArrayFromImage(img_B)
                data = np.zeros((arr[0].shape[0], arr[0].shape[1], in_channels))
                for ba in range(in_channels):
                    data[:, :, ba] = arr[ba]
                del img_A
                img_A = data
                img_B = np.array(img_B)
                """
                tempa = np.array(img_A)                
                for qr in range(in_channels):
                    temp = np.squeeze(tempa[:, :, qr])
                    temp[temp < 0] = 0
                    den = (np.max(temp.flatten()) - np.min(temp.flatten()))
                    if (den == 0):
                        den = 1
                    temp = (2 * (temp - np.min(temp.flatten())) / den) - 1
                    tempa[:, :, qr] = temp
                    del temp
                img_A = tempa  # *255
                del tempa
                temp = img_B
                temp[temp < 0] = 0
                den = (np.max(temp.flatten()) - np.min(temp.flatten()))
                if (den == 0):
                    den = 1
                temp = (2 * (temp - np.min(temp.flatten())) / den) - 1
                img_B = temp  # *255
                del temp
                """
                img_B = np.expand_dims(img_B, axis=-1)
            img_A = np.expand_dims(img_A, axis=0)
            img_B = np.expand_dims(img_B, axis=0)

            yield img_A, img_B

    def imread(self, path):
        return imageio.imread(path).astype(np.float)

