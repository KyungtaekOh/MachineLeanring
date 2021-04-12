from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import shutil
import math
import csv

# ['loss', 's_out_loss', 'c_out_loss', 's_out_acc', 'c_out_acc', 
#  'val_loss', 'val_s_out_loss', 'val_c_out_loss', 'val_s_out_acc', 'val_c_out_acc']
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, save_file_dir=None, target='val_loss'):
        self.patience = patience
        self.restore_count = 0
        self.wait = 0
        self.epoch = 0
        self.loss = np.Inf
        self.save_file_dir = save_file_dir
        self.best_weight = None
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        val_loss = logs.get(self.target)
        if(np.less(val_loss, self.loss)):
            print('## Load weight with Val_loss :', val_loss, '- Improve :', val_loss-self.loss)
            self.wait = 0
            self.loss = val_loss
            self.best_weight = self.model.get_weights()
            self.model.save_weights(self.save_file_dir)
        else:
            print("## Current val_loss =", val_loss, "less than previous val_loss!")
            self.wait += 1
            if(self.wait >= self.patience):
                self.restore_count += 1
                if(self.restore_count >= 5):
                    self.model.stop_training = True
                else:
                    print("## Restoring model weights from the end of the best epoch", val_loss)
                    self.model.set_weights(self.best_weight)

    def on_train_end(self, logs=None):
        print("## Stop training at epoch:{}_restoring{}".format(self.epoch, self.restore_count))


class DataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, data_list, batch_size, shuffle=True, mask_bool=False):
        self.mask_bool = mask_bool
        self.img_dir = img_dir
        if self.mask_bool:
            self.mask_dir = mask_dir
        self.data_list = data_list[:8201]
        self.batch_size = batch_size
        self.n = len(data_list)
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return math.ceil(len(self.data_list)/self.batch_size)

    def __getitem__(self, index):
        data_batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        xs, ys = self.__data_generation(data_batch)

        return xs, ys

    def __data_generation(self, data_list_temp):
        x = [[] for i in range(len(data_list_temp))]
        if self.mask_bool:
            y = dict(s_out=[], c_out=[])
        else:    
            y = dict(c_out=[])
        
        for idx, (img, msk, grade) in enumerate(data_list_temp):
            image = np.array(Image.open(self.img_dir + '/' + img))
            if self.mask_bool:
                mask  = np.array(Image.open(self.mask_dir + '/' + msk))
                mask  = mask.reshape(mask.shape+(1,))
                mask  = mask // 255
                y['s_out'].append(mask)
            x[idx] = image / 255.
            y['c_out'].append(int(grade)-1)
        x = np.array(x, dtype=np.float64)
        if self.mask_bool:
            y['s_out'] = np.array(y['s_out'])
        class_out = tf.keras.utils.to_categorical(y['c_out'], num_classes=9, dtype='float32')
        y['c_out'] = class_out

        return x, y

class Prediction:
    def __init__(self, data_list, img_dir, mask_dir, output_dir, model, save_bool=False):
        self.data_list = data_list[:]
        self.img_dir =  img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.save_bool = save_bool
        self.img_list, self.mask_list, self.og_class = self.__read_csv(self.data_list)
        self.model = model          # after load_weights()
        self.n = len(self.data_list)
        self.threshold_value = 0.7
        self.iou, self.mae = 0, 0
        self.p = []
        self.s_pred, self.c_pred = self.__prediction(self.model)
        # self.s_pred, self.c_pred = self.prediction_for_crop_test(self.model)

    def __prediction(self, model):
        s_pred_list, c_pred_list = [], []
        sum_of_iou = 0
        pred = model.predict(self.img_list, batch_size=4, verbose=1)
        self.p = pred
        for idx, (img_name, mask_name, grade) in tqdm(enumerate(self.data_list)):
            ### Segmentation Prediction
            s_pre = pred[0][idx].reshape(self.mask_list[idx].shape[0], self.mask_list[idx].shape[1])
            s_pre = np.where(s_pre > self.threshold_value, 255, 0).astype(np.uint8)
            s_pred_list.append(s_pre)

            ### Classification Predcition
            c_pre = pred[1][idx].argmax()
            c_pred_list.append(c_pre + 1)

            ### Save Image and Prediction
            if self.save_bool:
                if not (self.__save_pred_mask(s_pre, img_name)):
                    print("Something wrong with save process!")
            sum_of_iou += self.__cls_iou(self.mask_list[idx], s_pre/255)
        self.iou = sum_of_iou/self.n
        self.mae = self.__cls_mae(self.og_class, pred[1])
        
        return np.array(s_pred_list), np.array(c_pred_list)

    def prediction_for_crop_test(self, model):
        s_pred_list, c_pred_list = [], []
        sum_of_iou = 0
        pred = model.predict(self.img_list, batch_size=1, verbose=1)
        self.p = pred
        for idx, (img_name, mask_name, grade) in tqdm(enumerate(self.data_list)):
            pre = (pred[idx] * 255).astype(np.uint8)
            self.__save_pred_mask(pre, img_name)
        return np.array(s_pred_list), np.array(c_pred_list)

    def __cls_mae(self, true, pred):
        true = tf.argmax(true, -1)
        pred = tf.argmax(pred, -1)

        true = tf.cast(true, dtype=tf.float32)
        pred = tf.cast(pred, dtype=tf.float32)

        mae = tf.keras.losses.MeanAbsoluteError()
        return mae(true, pred)
    

    def __cls_iou(self, true, pred):
        intersection = tf.reduce_sum((true*pred).astype(np.int32))
        union = tf.reduce_sum(np.where(true+pred > 0, 1, 0).astype(np.int32))

        return intersection / union

    def __save_pred_mask(self, pred, file_name):
        img = Image.fromarray(pred)
        fname = file_name.split('.')
        img.save(self.output_dir + fname[0]+'_PRED.'+fname[1])

        shutil.copyfile(self.img_dir+file_name, self.output_dir+file_name)

        return True

    def __read_csv(self, name_list):
        img_list, mask_list, class_list = [], [], []
        print("Start read csv file.")
        for img_name, mask_name, grade in tqdm(name_list):
            img = np.array(Image.open(self.img_dir+img_name))/255.
            img_list.append(img)
            mask = np.array(Image.open(self.mask_dir+mask_name))/255.
            mask_list.append(mask)
            class_list.append(int(grade)-1)
        onehot_class = tf.keras.utils.to_categorical(np.array(class_list), num_classes=9, dtype='float32')
        
        return np.array(img_list), np.array(mask_list), onehot_class


def cls_mae(true, pred):
    true = tf.argmax(true, -1)
    pred = tf.argmax(pred, -1)
#
    true = tf.cast(true, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)
#
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(true, pred)

def cls_mae_cce(a=0.8):
    
    def func(true, pred):
        cce = tf.keras.losses.CategoricalCrossentropy()    
        return a * cce(true, pred) + (1 - a) * cls_mae(true, pred)

    return func


def split_csv(directory, csv_file, valid_ratio):
    data_list = []
    f = open(csv_file, 'r')
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
        data_list.append(line)

    random.shuffle(data_list)
    num_of_valid = math.ceil(len(data_list) * valid_ratio)
    train_data_list = np.array(data_list[num_of_valid:])
    valid_data_list = np.array(data_list[:num_of_valid])

    f = open(directory + '/splited_train.txt', 'w', newline='')
    writer = csv.writer(f, delimiter='\t')
    for line in train_data_list:
        writer.writerow(line)
    f.close()

    f = open(directory + '/splited_valid.txt', 'w', newline='')
    writer = csv.writer(f, delimiter='\t')
    for line in valid_data_list:
        writer.writerow(line)
    f.close()

    return True


def csv2arr(file_path, desired_cols=[0, 1, 2], encoding='cp949'):
    arr = []
    f = open(file_path, 'r', encoding=encoding)
    rdr = csv.reader(f, delimiter='\t')
    for line in rdr:
        temp = []
        for i in desired_cols: 
            temp.append(line[i])
        arr.append(temp)

    return arr

