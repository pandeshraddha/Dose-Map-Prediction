from keras.models import load_model
from datetime import datetime
from glob import glob
import os
from pix2pix_model import define_discriminator, define_generator, define_gan, train
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# define input shape based on the loaded dataset
image_shape = (256,256,6)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)
# define the composite model

#d_model = load_model('dmodel_40.h5')
opt = Adam(lr=0.0002, beta_1=0.5)
d_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
#g_model = load_model('gmodel_40.h5')
g_model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[1, 100])
gan_model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[1, 100])

start1 = datetime.now()
train(d_model, g_model, gan_model)
# Reports parameters for each batch (total 1096) for each epoch.
# For 10 epochs we should see 10960
stop1 = datetime.now()
# Execution time of the model
execution_time = stop1 - start1
print("Execution time is: ", execution_time)