import os
import csv
import json
import tools
import model
import model_v2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

directory = r'/home/cvlab09/kyung-taek/cnn/seg_grad'
img_dir = r'/home/cvlab09/kyung-taek/cnn/data'
csv_dir = r'/home/cvlab09/kyung-taek/cnn/train.txt'

batch = 4
epoch = 100

param = {'trial'        :   6  ,          ########## Modify before train ##########
         'model_name'   :   'Unet_V4_ver2',
         'batch'        :   batch,
         'epoch'        :   epoch,
         'lr'           :   1e-4,
         's_loss'       :   'binary_crossentropy',
         'c_loss'       :   'mae_cce_loss(0.7)'}

save_file_name = "{}_t{}.h5".format(
                    param['model_name'], param['trial'])


#########################################
# Split Train & Valid from Raw CSV File #
#########################################
from os.path import isfile
if not (isfile(directory + '/splited_train.txt') and 
        isfile(directory + '/splited_valid.txt')):
    if(tools.split_csv(directory, csv_dir, 0.3) == True):
        print('Successfully split!')
    else:
        print('There is a problem in the split process!')
else:
    print('Already in directroy!')


train_data_list = tools.csv2arr(r'/home/cvlab09/kyung-taek/cnn/segmentation/splited_train.txt',
                                desired_cols=[0, 1, 2],
                                encoding='utf-8')


valid_data_list = tools.csv2arr(r'/home/cvlab09/kyung-taek/cnn/segmentation/splited_valid.txt',
                                desired_cols=[0, 1, 2],
                                encoding='utf-8')


##################################
# Generator                      #
# x : Images - b16, RGB, 0~1     #
# y : 'Mask' - b16, Gray, 0 or 1 #
#     'Grade' - b16, class, 0~9  #
##################################
train_generator = tools.DataGenerator(img_dir=img_dir+'/image',
                                      mask_dir=img_dir+'/mask',
                                      data_list=train_data_list,
                                      batch_size=batch,
                                      mask_bool=True)
valid_generator = tools.DataGenerator(img_dir=img_dir+'/image',
                                      mask_dir=img_dir+'/mask',
                                      data_list=valid_data_list,
                                      batch_size=batch,
                                      mask_bool=True)


step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = valid_generator.n // valid_generator.batch_size


#######################################
# Model : U-Net + Inception_resnet_v4 #
#######################################
model = model.getModel()

adam = tf.keras.optimizers.Adam(lr=param['lr'])
mae_cce_loss = tools.cls_mae_cce(a=0.7)
model.compile(
    optimizer = adam,
    loss={'s_out':param['s_loss'], 'c_out':mae_cce_loss},
    metrics={'s_out':'acc', "c_out":'acc'}
)

history_file_name = "{}_t{}_history.txt".format(param['model_name'], param['trial'])
history = tf.keras.callbacks.CSVLogger('/home/cvlab09/kyung-taek/cnn/history/'+history_file_name, separator="\t", append=True)

custom_callback = tools.CustomCallback(patience=10, 
                                       save_file_dir=r'/home/cvlab09/kyung-taek/cnn/saved_weight/'+save_file_name,
                                       target='val_loss')

param_file_name = "{}_t{}_param.json".format(param['model_name'], param['trial'])
param_file = open("/home/cvlab09/kyung-taek/cnn/history/" + param_file_name, "w")
json.dump(param, param_file, separators=(',\n', '\t:\t'))
param_file.close()

model.fit(train_generator,
          steps_per_epoch=step_size_train,
          validation_data=valid_generator,
          validation_steps=step_size_valid,
          epochs=epoch,
          verbose=1,
          callbacks=[custom_callback, history])
