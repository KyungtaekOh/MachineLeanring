import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import numpy as np
import loss
import os
import xmltodict
import random
import math
from tqdm import tqdm

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
        print("Epoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

LR_SCHEDULE = [(0, 0.01),
               (50, 0.01),
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
    def __init__(self, info_list, batch_size, img_dir, grid_size, num_bboxes, num_classes, shuffle=True):
        self.data_list = info_list
        self.n = len(self.data_list)
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.reshape_size = (448, 448)
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
            image = np.array(Image.open(os.path.join(self.img_dir, img_name)).resize(self.reshape_size))

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

class Prediction:
    def __init__(self, test_gene, output_dir, model, class_arr):
        self.test_gene = test_gene
        self.n = self.test_gene.n
        self.batch_size = self.test_gene.batch_size
        self.data_list = self.test_gene.data_list
        self.img_dir = self.test_gene.img_dir
        self.output_dir = output_dir
        self.model = model
        self.class_arr = class_arr
        self.__prediction()

    def __prediction(self):
        boxes = []
        pred = self.model.predict(self.test_gene, batch_size=self.batch_size, verbose=1) # ?*7*7*30
        for idx in tqdm(range(self.n)): # Each Image unit
            y = pred[idx,:,:,:] # 7*7*30
            bboxes, class_score = y[:,:,:10], y[:,:,10:] # 7*7*10, 7*7*20
            box1_conf = K.expand_dims(bboxes[:,:,4], axis=2) # 7*7*1
            box2_conf = K.expand_dims(bboxes[:,:,9], axis=2) # 7*7*1
            box1_class_score = tf.where((class_score * box1_conf)<0.2, 0)
            box2_class_score = tf.where((class_score * box2_conf)<0.2, 0)
            box1_class_score = K.reshape(box1_class_score, (7*7,20))    # 49*20
            box2_class_score = K.reshape(box2_class_score, (7*7,20))    # 49*20
            boxes_class_score = K.concatenate([box1_class_score, box2_class_score], axis=0) # 98*20
            length = boxes_class_score.shape[0] # 98
            for i in range(length): # Class unit
                bbox = K.argmax(boxes_class_score[:, i]).numpy()
                bbox_class = K.argmax(boxes_class_score[bbox, :]).numpy()
                bbox_conf = boxes_class_score[bbox, bbox_class].numpy()
                if bbox_conf > 0.3:
                    cell_idx = math.ceil(bbox / 2)
                    bbox_idx = bbox % 2
                    coord = bboxes[cell, 4*(bbox_idx):4*(bbox_idx+1)]
                    self.__draw_box(self.data_list[idx], coord.numpy())
                    continue

    def __draw_box(self, data_info, coord):
        image = np.array(Image.open(os.path.join(data_info['img_name'], img_name)))
        im_w, im_h = data_info['img_size']
        x, y, w, h = coord
        x1, x2 = math.ceil((x-w/2)*im_w), math.ceil((x+w/2)*im_w)
        y1, y1 = math.ceil((y-h/2)*im_h), math.ceil((y+h/2)*im_h)

        draw = ImageDraw.Draw(image)
        draw.rectangle(((x1, y1), (x2, y2)), outline(0,255,0), width=2)

        image.save(os.path.join(self.output_dir,data_info['img_name']))
        return True

def get_info(annotation_f, classes):
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