import tensorflow as tf
import utils, yolo, loss
import json
import os
from train import param, test_list, classes

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

images = '/home/cvlab09/kyung-taek/detection/YOLOv1_darknet/VOCdevkit/VOC2007/JPEGImages'

# test_generator = utils.DataGenerator(test_list, 4, images, param['grid_size'], param['num_bboxes'], param['num_classes'])
test_generator = utils.DataGenerator(test_list[:16], 4, images, param['grid_size'], param['num_bboxes'], param['num_classes'])
output_dir = r'/home/cvlab09/kyung-taek/detection/YOLOv1/output'

model = yolo.YOLOv1()
weight = r'/home/cvlab09/kyung-taek/detection/YOLOv1/weights/YOLOv1_t1.h5'
model.load_weights(weight)

pred = utils.Prediction(test_generator, output_dir, model, classes)


