import tensorflow as tf
import os
import utils, yolo, loss

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

annotation = '/home/cvlab09/kyung-taek/detection/YOLOv1_darknet/VOCdevkit/VOC2007/Annotations'
images = '/home/cvlab09/kyung-taek/detection/YOLOv1_darknet/VOCdevkit/VOC2007/JPEGImages'

classes = ['person', # Person
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', # Animal
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', # Vehicle
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor' # Indoor
           ]

num_classes = len(classes)
grid_size = 7
num_bboxes = 2

info_list = utils.get_info(annotation, classes)
print(len(info_list))
train_list = info_list[:4000]
valid_list = info_list[4000:]

# print(train_valid_list[:6])
x, y = utils.data_generation(train_list[:16], images)






