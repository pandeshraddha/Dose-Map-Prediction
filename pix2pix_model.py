"""
pix2pix GAN model
Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/
I seriously urge everyone to foloow his blogs and get enlightened.
I am adapting his code to various applications but original credit goes to Jason.
    Original paper: https://arxiv.org/pdf/1611.07004.pdf
    Github for original paper: https://phillipi.github.io/pix2pix/

Generator:
The encoder-decoder architecture consists of:
encoder:
C64-C128-C256-C512-C512-C512-C512-C512
decoder:
CD512-CD512-CD512-C512-C256-C128-C64
Discriminator
C64-C128-C256-C512
After the last layer, a convolution is applied to map to
a 1-dimensional output, followed by a Sigmoid function.
"""
#
import math
import imageio
import tensorflow as tf
import SimpleITK as sitk
import random
from scipy.io import savemat
import matplotlib
from numpy import vstack
import os
import numpy as np
from glob import glob
from dataloading import DataLoader
from numpy import zeros
from numpy import ones
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU, ReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
import matplotlib
import gc
import tkinter
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
n_channels = 6
out_channels = 1
img_res=(256, 256)
#############################################################################
# Define generator, discriminator, gan and other helper functions
# We will use functional way of defining model and not sequential
# as we have multiple inputs; both images and corresponding labels.
########################################################################

# Since pix2pix is a conditional GAN, it takes 2 inputs - image and corresponding label
# For pix2pix the label will be another image.

# define the standalone discriminator model
# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.
# Think of discriminator as a binary classifier that is classifying images as real/fake.

# From the paper C64-C128-C256-C512
# After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.

def my_validation(t_g_model,epoch,batch_i):
    n_channels = 6
    path = 'F:\\Desktop\\Documeents\\GAN\\download_data\\BRATS2015_Training\\BRATS2015_Training\\my_bcmb\\global_scalingrtgt20\\test\\T1\\'
    pas = os.listdir(path)
    pas.sort(key=str.lower)
    batch_size = 1
    file_num = random.sample(range(689,727),3)
    imgq_B = []
    imgq_A = []
    genq_B = []
    OT = []
    patha = 'F:\\Desktop\\Documeents\\GAN\\download_data\\BRATS2015_Training\\BRATS2015_Training\\my_bcmb\\global_scalingrtgt20\\test\\'
    for q in range(3):
        num = file_num[q]
        batch = pas[num * batch_size:(num + 1) * batch_size]
        imgs_A, imgs_B = [], []
        for imgx in batch:
            p = os.listdir(patha)
            p.sort(key=str.lower)
            arr = []
            for i in range(len(p)):
                if ((i == 0) or (i == 1) or (i == 7) or (i == 8) or (i == 9) or (i == 10)):
                    img_A = sitk.ReadImage((patha +'/'+ p[i] + '/' + imgx))
                    arr.append(sitk.GetArrayFromImage(img_A))
                elif (i == 2):
                    img_B = sitk.ReadImage((patha +'/'+ p[i] + '/' + imgx))
                    img_B = sitk.GetArrayFromImage(img_B)
            data = np.zeros((arr[0].shape[0], arr[0].shape[1], n_channels))
            for ba in range(n_channels):
                data[:, :, ba] = arr[ba]
            del img_A
            img_A = data
            img_B = np.array(img_B)
            """
            tempa = np.array(img_A)
            for qr in range(n_channels):
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
        gen_image = t_g_model.predict(img_A)
        imgq_B.append(np.squeeze(img_B))
        genq_B.append(np.squeeze(gen_image))
        imgq_A.append(np.squeeze(img_A[:, :, :,0]))  # T1w imgq_A[2] = np.squeeze(img_A[:, :, :,1])
    #seg_results = {'reference': imgq_A, 'groundtruth': imgq_B, 'generated': genq_B, 'tumor':OT}
    #file_name = "./%s%d%d%s" % ('result-',batch_i,epoch,'.mat') #fold[ID] + '.mat'
    #newpath = 'F:\\Desktop\\Documeents\\GAN\\forwardpathv1\\imagesb\\matfiles\\'
    #savemat(os.path.join(newpath, file_name), seg_results)
    sample_images(imgq_A,genq_B,imgq_B, epoch, batch_i)

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)  # As described in the original paper

    # source image input
    in_src_image = Input(shape=image_shape)  # Image we want to convert to another image
    # target image input
    output_shape = (256, 256, out_channels)
    in_target_image = Input(shape=output_shape)  # Image we want to generate after training.

    # concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(out_channels, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    # The model is trained with a batch size of one image and Adam opt.
    # with a small learning rate and 0.5 beta.
    # The loss for the discriminator is weighted by 50% for each model update.

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    model.summary()
    return model




# disc_model = define_discriminator((256,256,3))
# plot_model(disc_model, to_file='disc_model.png', show_shapes=True)

##############################
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model - U-net
def define_generator(image_shape=(256, 256, n_channels)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model: CD512-CD512-CD512-C512-C256-C128-C64
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(out_channels, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
        d7)  # Modified
    out_image = Activation('tanh')(g)  # Generates images in the range -1 to 1. So change inputs also to -1 to 1
    # define model
    model = Model(in_image, out_image)
    return model


# gen_model = define_generator((256,256,3))
# plot_model(gen_model, to_file='gen_model.png', show_shapes=True)


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False  # Descriminator layers set to untrainable in the combined GAN but
            # standalone descriminator will be trainable.

    # define the source image
    in_src = Input(shape=image_shape)
    # suppy the image as input to the generator
    gen_out = g_model(in_src)
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)

    # Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    # Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[1, 100])
    return model
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, n_channels))
    return [X1, X2], y
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, n_channels))
    return X, y
def sample_images(imgs_B, fake_A, imgs_A,epoch,batch_i):
    #os.makedirs('./imagesb/%s' % 'result', exist_ok=True)
    r, c = 3, 3
    #imgs_A, imgs_B = (DataLoader.load_data(batch_size=3))
    #fake_A = g_model.predict(np.asarray(imgs_B))
    gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
    # Rescale images 0 - 1
    gen_imgs = (gen_imgs + 1)
    titles = ['Condition', 'Generated', 'Target']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("./imagesb/%s/%d_%d.png" % ('result', epoch, batch_i))
    plt.close()
    plt.cla()
    plt.clf()
    del fig,axs, gen_imgs
    gc.collect()


# train pix2pix models
def train(d_model, g_model, gan_model,epochs=1000, batch_size=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    m_gloss = []
    m_dloss1=[]
    m_dloss2=[]
    for epoch in range(epochs):
        allgendata = []
        allgtdata   = []
        allrtmap = []
        allreferencemap = []
        for batch_i, (X_realA, X_realB) in enumerate(DataLoader.load_batch(batch_size)):
            y_real = ones((batch_size, n_patch, n_patch,1))
        # select a batch of real samples
            X_fakeB = g_model.predict( X_realA)
            allgendata.append(X_fakeB)
            allrtmap.append(np.squeeze(X_realA[:,:,:,2]))
            allreferencemap.append(np.squeeze(X_realA[:,:,:,0]))
            # create 'fake' class labels (0)
            y_fake = zeros((batch_size, n_patch, n_patch, 1))
            # update discriminator for real samples
            sdev = 0.1 * math.exp(-0.25*epoch)
            noiseA = np.random.normal(0,sdev,X_realA.shape)
            noiseB = np.random.normal(0,sdev,X_realB.shape)
            d_loss1 = d_model.train_on_batch([X_realA + noiseA, X_realB + noiseB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA + noiseA, X_fakeB + noiseB], y_fake)
            # update the generator
            g_loss,_,_= gan_model.train_on_batch(X_realA, [y_real, X_realB])
            allgtdata.append(X_realB)
            m_dloss1.append(d_loss1)
            m_dloss2.append(d_loss2)
            m_gloss.append(g_loss)
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f] %f' % (batch_i + 1, d_loss1, d_loss2, g_loss,(d_loss1+d_loss2)/2))
            file = open("F:\\Desktop\\Documeents\\GAN\\forwardpathv1testing\\python.txt", "w")
            file.write("%s = %s\n" % ("total_gen_loss", m_gloss))
            file.write("%s = %s\n" % ("discriminator_fake_loss", m_dloss1))
            file.write("%s = %s\n" % ("discriminator_real_loss", m_dloss2))
            file.close()
            #if batch_i % 1000 == 0:
                #my_validation(g_model,epoch,batch_i)
        if (epoch % 20 == 0):
            g_model.save('gmodel_%d' % (epoch))
            d_model.save('dmodel_%d' % (epoch))
            seg_results = {'GT': allgtdata, 'generated': allgendata, 'rt': allrtmap, 'adc': allreferencemap}
            file_name = "./%s%d%s" % ('result-', epoch, '.mat')  # fold[ID] + '.mat'
            newpath = 'F:\\Desktop\\Documeents\\GAN\\forwardpathv1testing\\imagesb\\matfiles\\'
            savemat(os.path.join(newpath, file_name), seg_results)
