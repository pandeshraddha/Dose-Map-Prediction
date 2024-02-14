"""
from keras.models import load_model
import os
import numpy as np
import SimpleITK as sitk
import random
from tensorflow.keras.optimizers import Adam
from scipy.io import savemat
from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
def my_validation(t_g_model):
    n_channels = 6
    generatorloss = []
    path = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\trainingdata\\test\\T1\\'
    pas = os.listdir(path)
    pas.sort(key=str.lower)
    patha = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\trainingdata\\test\\'
    for q in range(len(pas)):
        p = os.listdir(patha)
        p.sort(key=str.lower)
        arr = []
        for i in range(len(p)):
            if ((i == 0) or (i == 1) or (i == 7) or (i == 8) or (i == 9) or (i == 10)):
                img_A = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                arr.append(sitk.GetArrayFromImage(img_A))
            elif (i == 13):
                img_B = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                img_B = sitk.GetArrayFromImage(img_B)
            elif (i == 3):
                img_C = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                img_C = sitk.GetArrayFromImage(img_C)
            elif (i == 12):
                img_D = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                img_D = sitk.GetArrayFromImage(img_D)
        data = np.zeros((arr[0].shape[0], arr[0].shape[1], n_channels))
        for ba in range(n_channels):
            data[:, :, ba] = arr[ba]
        del img_A
        img_A = data
        img_B = np.array(img_B)
        img_A = np.expand_dims(img_A, axis=0)
        gen_image = t_g_model.predict(img_A)
        #tere = t_g_model.evaluate(img_A, gen_image)
        #generatorloss.append(tere)
        seg_results = {'reference': np.squeeze(img_A[:, :, :, 1]), 'groundtruth': np.squeeze(img_B),'generated': np.squeeze(gen_image),'RT': np.squeeze(img_A[:, :, :, 2]),'originalpostRT':img_C,'tumor':img_D}
        file_name = "./%s%s" % (pas[q].removesuffix('.nii'), '.mat')  # fold[ID] + '.mat'
        newpath = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\testingmodel\\imagesb\\matfiles\\flair\\'
        savemat(os.path.join(newpath, file_name), seg_results)
        #file = open("F:\\Desktop\\Documeents\\GAN\\forwardpathv1testing\\python.txt", "w")
        #file.write("%s = %s\n" % ("test_gen_loss", generatorloss))
        #file.close()

image_shape = (256,256,6)
#opt = Adam(lr=0.0002, beta_1=0.5)
#d_model = define_discriminator(image_shape)
#g_model = define_generator(image_shape)
#gan_model = define_gan(g_model, d_model, image_shape)
g_model = load_model('F:\\Desktop\\Documeents\\GAN\\trainedmodels\\flair\\gmodel_175')
my_validation(g_model)
#d_model = load_model('F:\\Desktop\\Documeents\\GAN\\forwardpathv1testing\\dmodel_350.h5')
#d_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

#g_model.compile(loss=['binary_crossentropy', 'mae'],
                  #optimizer=opt, loss_weights=[1, 100])
"""

############################## Predicting Inversemodel
from keras.models import load_model
import os
import numpy as np
import SimpleITK as sitk
import random
from tensorflow.keras.optimizers import Adam
from scipy.io import savemat
from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
def my_validation(t_g_model):
    n_channels = 10
    generatorloss = []
    path = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\trainingdata\\test\\T1\\'
    pas = os.listdir(path)
    pas.sort(key=str.lower)
    patha = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\trainingdata\\test\\'
    for q in range(len(pas)):
        p = os.listdir(patha)
        p.sort(key=str.lower)
        arr = []
        for i in range(len(p)):
            if ((i == 0) or (i == 1) or (i == 8) or (i == 9) or (i == 10) or (i == 17) or (i == 18) or (i == 19) or (i == 20) or (i == 21)):
                img_A = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                arr.append(sitk.GetArrayFromImage(img_A))
            elif (i == 7):
                img_B = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                img_B = sitk.GetArrayFromImage(img_B)
            elif (i == 12):
                img_C = sitk.ReadImage((patha + '/' + p[i] + '/' + pas[q]))
                img_C = sitk.GetArrayFromImage(img_C)
        data = np.zeros((arr[0].shape[0], arr[0].shape[1], n_channels))
        for ba in range(n_channels):
            data[:, :, ba] = arr[ba]
        del img_A
        img_A = data
        img_B = np.array(img_B)
        img_A = np.expand_dims(img_A, axis=0)
        gen_image = t_g_model.predict(img_A)
        #tere = t_g_model.evaluate(img_A, gen_image)
        #generatorloss.append(tere)
        seg_results = {'referencea': np.squeeze(img_A[:, :, :, 0]),'referenceb': np.squeeze(img_A[:, :, :, 1]) ,'referencec': np.squeeze(img_A[:, :, :, 2]),'referenced': np.squeeze(img_A[:, :, :, 3]),'referencee': np.squeeze(img_A[:, :, :, 4]),'groundtruth': np.squeeze(img_B),'generated': np.squeeze(gen_image),'postrta':np.squeeze(img_A[:, :, :, 5]),'postrtb':np.squeeze(img_A[:, :, :, 6]),'postrtc':np.squeeze(img_A[:, :, :, 7]),'postrtd':np.squeeze(img_A[:, :, :, 8]),'postrte':np.squeeze(img_A[:, :, :, 9]),'tumora':img_C}
        file_name = "./%s%s" % (pas[q].removesuffix('.nii'), '.mat')  # fold[ID] + '.mat'
        newpath = 'F:\\Desktop\\Documeents\\redoinganalysisfor1192022bcmb1\\testingmodel\\imagesb\\matfiles\\inversemodel10inputpregtvsm\\'
        savemat(os.path.join(newpath, file_name), seg_results)
        #file = open("F:\\Desktop\\Documeents\\GAN\\forwardpathv1testing\\python.txt", "w")
        #file.write("%s = %s\n" % ("test_gen_loss", generatorloss))
        #file.close()

image_shape = (256,256,10)
g_model = load_model('M:\\dept\IRAT_Research\\Shraddha_Pandey\\inversemodel\\output 20220913\\inversemodel\\gmodel_675')
my_validation(g_model)
