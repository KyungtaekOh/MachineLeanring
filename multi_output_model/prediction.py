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

data_list = tools.csv2arr(file_path=r'test.txt',
                          desired_cols=[0, 1, 2])

img_dir = r'test/'
mask_dir = r'test_GT/'
output_dir = r'output_crop/'

model = model.getModel()

pred = tools.Prediction(data_list, img_dir, mask_dir, output_dir, model, save_bool=False)
print('MIoU :', pred.iou.numpy())
print('MAE :', pred.mae.numpy())
