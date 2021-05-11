import os
import tools
import random
import model_Unet
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

valid_data_list = tools.csv2arr(r'test.txt',
                                [0, 1])

img_dir = r'test/'
mask_dir = r'test_GT/'
output_dir = r'test_results/'


model = model_Unet.getUnet()
weight = r'saved_weight/Unet_e100_b16_t2.h5'
model.load_weights(weight)

pred = tools.Prediction(valid_data_list, img_dir, mask_dir, output_dir, model)
print(pred.iou.numpy())

