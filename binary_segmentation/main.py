"""
Goal : IoU > 95%
"""

import os
import tools
import model_Unet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


directory = r'segmentation'
img_dir = r'data'
csv_file = r'train.txt'

model_name = 'Unet'
batch = 8
epoch = 500
trial = 4
save_file_name = "{}_e{}_b{}_t{}.h5".format(
                    model_name, epoch, batch, trial)


#########################################
# Split Train & Valid from Raw CSV File #
#########################################
from os.path import isfile
if not (isfile(directory + '/splited_train.txt') and 
        isfile(directory + '/splited_valid.txt')):
    if(tools.split_csv(directory, csv_file, 0.3) == True):
        print('Successfully split!')
    else:
        print('There is a problem in the split process!')
else:
    print('Already in directroy!')



train_data_list = tools.csv2arr(r'splited_train.txt',
                                [0, 1])


valid_data_list = tools.csv2arr(r'splited_valid.txt',
                                [0, 1])


##################################
# Generator                      #
# x : Images - #16, RGB, 0~1     #
# y : Mask - #16, Gray, 0 or 1   #
##################################
train_generator = tools.DataGenerator(img_dir=img_dir+'/image',
                                      mask_dir=img_dir+'/mask',
                                      data_list=train_data_list,
                                      batch_size=batch)
valid_generator = tools.DataGenerator(img_dir=img_dir+'/image',
                                      mask_dir=img_dir+'/mask',
                                      data_list=valid_data_list,
                                      batch_size=batch)
step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = valid_generator.n // valid_generator.batch_size


#################
# Model : U-Net #
#################
model = model_Unet.getUnet()

adam = keras.optimizers.Adam(lr=1e-6)
model.compile(
    optimizer = adam,
    loss='binary_crossentropy',
    metrics=['acc']
)
checkpoint = ModelCheckpoint('saved_weight/'+save_file_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
myCallback = tools.myCallback()
history_file_name = "{}_e{}_b{}_t{}_history.txt".format(model_name, epoch, batch, trial)
history = keras.callbacks.CSVLogger('history/'+history_file_name, separator="\t", append=False)

model.fit(train_generator,
          steps_per_epoch=step_size_train,
          validation_data=valid_generator,
          validation_steps=step_size_valid,
          epochs=epoch,
          callbacks=[checkpoint, history]
          )
            
