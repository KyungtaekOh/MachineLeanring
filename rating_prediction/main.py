"""
tensorflow version : 2.4 or 2.4.1
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import tools
import model_resnet
import random
import math
#
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#
path = r'/home/cvlab09/kyung-taek/cnn/'
img_dir = 'data/image/'
csv_file = 'train.txt'
#
model_name = 'Resent50'
batch = 16          # 16 proper
epoch = 10         # m 100, M 1000
trial = 2
save_file_name = "{}_e{}_b{}_t{}.h5".format(model_name, epoch, batch, trial)



#########################################
# Split Train & Valid from Raw CSV File #
#########################################
from os.path import isfile
if not (isfile(r'/home/cvlab09/kyung-taek/cnn/splited_train.txt') and 
        isfile(r'/home/cvlab09/kyung-taek/cnn/splited_valid.txt')):
    if(tools.split_csv(path + csv_file, 0.2, [0, 2]) == True):
        print('Successfully split!')
    else:
        print('There is a problem in the split process!')
else:
    print('Already in directroy!')

train_data_list = tools.csv2arr(r'/home/cvlab09/kyung-taek/cnn/splited_train.txt')
valid_data_list = tools.csv2arr(r'/home/cvlab09/kyung-taek/cnn/splited_valid.txt')


#################################
# Generator                     #
# x : Images - #16, RGB, 0~1    #
# y : onehot label              #
#################################
train_generator = tools.DataGenerator(train_data_list, path, batch)
valid_generator = tools.DataGenerator(valid_data_list, path, batch)

step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size


#########
# Model #
#########
checkpoint = ModelCheckpoint(path+'saved_weight/'+save_file_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

import model_resnet
model = model_resnet.getResnet50()

adam = keras.optimizers.Adam(lr=0.1, decay=1e-4)
loss = cls_mae_cce(a=0.8)
model.compile(
    optimizer = adam,
    loss= loss,
    metrics=['acc', tools.cls_mae]
)

redueceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
history_file_name = "{}_e{}_b{}_t{}_history.txt".format(model_name, epoch, batch, trial)
history = keras.callbacks.CSVLogger(path+'history/'+history_file_name, separator="\t", append=False)

model.fit(train_generator,
          steps_per_epoch=step_size_train,
          validation_data=valid_generator,
          validation_steps=step_size_valid,
          epochs=epoch,
          callbacks=[checkpoint, redueceLR, history])

