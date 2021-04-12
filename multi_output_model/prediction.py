import os
import tools
import random
import model_v2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

data_list = tools.csv2arr(file_path=r'/home/cvlab09/kyung-taek/cnn/segmentation/test.txt',
                          desired_cols=[0, 1, 2])

img_dir = r'/home/cvlab09/kyung-taek/cnn/segmentation/test/'
mask_dir = r'/home/cvlab09/kyung-taek/cnn/segmentation/test_GT/'
output_dir = r'output_crop/'

model = model.getModel()
# model = model_v2.getModel_v2()
# weight = r'/home/cvlab09/kyung-taek/cnn/saved_weight/Unet_V4_ver2_t5.h5'
# model.load_weights(weight)

# crop_out = model.layers[33]     # crop layer
# features_list = [layer.output for layer in model.layers[:34]]
# model2 = tf.keras.Model(inputs = model.input, outputs = features_list[-1])
# pred = tools.Prediction(data_list, img_dir, mask_dir, output_dir, model, save_bool=True)

pred = tools.Prediction(data_list, img_dir, mask_dir, output_dir, model, save_bool=False)
print('MIoU :', pred.iou.numpy())
print('MAE :', pred.mae.numpy())