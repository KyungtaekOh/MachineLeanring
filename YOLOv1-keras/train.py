import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
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

info_list = utils.get_info(annotation, classes) # 6045
batch_size = 4
epoch = 100
param = {'trial'        :   1  ,          ########## Modify before train ##########
         'model_name'   :   'YOLOv1',
         'grid_size'    :   7,
         'num_bboxes'   :   2,
         'num_classes'  :   len(classes),
         'batch'        :   batch_size,
         'epoch'        :   epoch,
         'trn_val_len'  :   5000,
         'trn_val_rate' :   0.3,
         'test_len'     :   len(info_list)-5000,
         'lr'           :   1e-4,
         'lr_schedule'  :   utils.LR_SCHEDULE,
         'momentum'     :   9,
         'weight_decay' :   5e-4,
         }

##############
# Split Data #
##############
train_valid_list = info_list[:5000]
test_list = info_list[5000:]

train_data_list, valid_data_list = utils.train_test_split(train_valid_list) # 3500, 1500
train_generator = utils.DataGenerator(train_data_list, batch_size, images, param['grid_size'], param['num_bboxes'], param['num_classes'])
valid_generator = utils.DataGenerator(valid_data_list, batch_size, images, param['grid_size'], param['num_bboxes'], param['num_classes'])
step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = valid_generator.n // valid_generator.batch_size

#############
# Callbacks #
#############
custom_scheduler = utils.custom_lr_scheduler(utils.lr_schedule)
save_file_name = "{}_t{}".format(param['model_name'], param['trial'])
history = tf.keras.callbacks.CSVLogger('/home/cvlab09/kyung-taek/detection/YOLOv1/history/'+save_file_name+'_history.txt', separator="\t", append=True)
checkpoint = ModelCheckpoint('/home/cvlab09/kyung-taek/detection/YOLOv1/weights/'+save_file_name+'.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

#########
# Model #
#########
model = yolo.YOLOv1()
model.summary()