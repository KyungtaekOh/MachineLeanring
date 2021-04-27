import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import xmltodict
import random
import math

class custom_lr_scheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(custom_lr_scheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

LR_SCHEDULE = [(0, 0.01),
               (75, 0.001),
               (105, 0.0001)]
def lr_schedule(epoch, lr):
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

class DataGenerator(Sequence):
    def __init__(self, info_list, batch_size, img_dir, grid_size, num_bboxes, num_classes, mode='train', shuffle=True):
        self.data_list = info_list
        self.n = len(self.data_list)
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.mode = mode
        if shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return math.ceil(len(self.data_list)/self.batch_size)

    def __getitem__(self, index):
        data_batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        xs, ys = self.__data_generation(data_batch)
        xs = np.array(xs)
        ys = np.array(ys)

        return xs, ys

    def __data_generation(self, batch_data_list):
        x = [[] for i in range(len(batch_data_list))]
        y = [[] for i in range(len(batch_data_list))]
        
        for idx, info in enumerate(batch_data_list):
            img_name = info['img_name']
            bboxes = info['bboxes']
            labels = info['labels']
            image = np.array(Image.open(os.path.join(self.img_dir, img_name)).resize((448, 448)))

            x[idx] = image / 255.
            y[idx] = self.__enc_to_tensor(bboxes, labels)
        return x, y

    def __enc_to_tensor(self, bboxes, labels):
        S = self.grid_size
        B = self.num_bboxes
        N = 5*B + self.num_classes # [box1, box2, classes]
        box_center = (bboxes[:,:2] + bboxes[:,2:]) / 2
        box_line_len = bboxes[:,2:] - bboxes[:,:2]
        output = np.zeros((S, S, N))

        for box in range(bboxes.shape[0]):
            center, line_len, label = box_center[box], box_line_len[box], labels[box]
            coord = np.floor(center * S)
            coord_i, coord_j = map(int, coord)
            output[coord_i, coord_j, :] = np.hstack([center*7, 
                                                     line_len*7, 
                                                     1,
                                                     center*7, 
                                                     line_len*7, 
                                                     1,
                                                     tf.keras.utils.to_categorical(label, 20)])
        return output


def get_info(annotation_f, classes, split_rate=0.3):
    an_dir = annotation_f
    result = []
    class_list = classes
    annot_list = [os.path.join(an_dir, ano) for ano in os.listdir(an_dir)]
    for idx, an in enumerate(annot_list):
        f = open(an)
        info = xmltodict.parse(f.read())['annotation']
        f.close()
        img_name = info['filename']
        img_size = np.asarray(tuple(map(int, info['size'].values()))[:2], np.int16)
        w, h = img_size
        box_object = info['object']
        labels, bboxes = [], []
        for obj in box_object:
            try:
                labels.append(class_list.index(obj['name']))
                bboxes.append(tuple(map(int, obj['bndbox'].values())))
            except:pass
        bboxes = np.asarray(bboxes, dtype=np.float64)
        try:
            bboxes[:, [0, 2]] /= w
            bboxes[:, [1, 3]] /= h
        except:pass

        if bboxes.shape[0]:
            result.append({'img_name':img_name,
                            'img_size':img_size,
                            'bboxes':bboxes,
                            'labels':labels})
    return result

def train_test_split(arr, split_rate=0.3):
    l = math.ceil(len(arr) * split_rate)
    return arr[l:], arr[:l]