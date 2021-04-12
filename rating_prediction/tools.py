import numpy as np
import pandas as pd
import random
import tensorflow as tf
import csv
import math
from PIL import Image
from tqdm import tqdm

def cls_mae(true, pred):
    true = tf.argmax(true, -1)
    pred = tf.argmax(pred, -1)
#
    true = tf.cast(true, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)
#
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(true, pred)

def cls_mae2(true, pred):
    pred = tf.argmax(pred, -1) + 1
    pred = tf.cast(pred, dtype=tf.float32)

    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(true, pred)

def cls_mae_cce(a=0.8):
    
    def func(true, pred):
        cce = tf.keras.losses.CategoricalCrossentropy()    
        return a * cce(true, pred) + (1 - a) * cls_mae(true, pred)

    return func

class DataGenerator:
    def __init__(self, data_list, path, batch_size, target_size=(512, 512)):
        self.data_list = self.__onehot(data_list)
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        self.index = 0
        self.n = len(data_list)
        self.shuffle = self.__shuffle_data()
    #
    def __iter__(self):
        return self
    #
    def __next__(self):
        batch_data = self.__get_batch_data()
        #
        x, y = [[] for i in range(self.batch_size)], [[] for i in range(self.batch_size)]
        for idx, (name, label) in enumerate(batch_data):
            image = np.array(Image.open(self.path + 'image/' + name))
            x[idx] = image / 255.
            y[idx] = label
        x = np.array(x)
        y = np.array(y)
        #
        return x, y
    #
    def __get_batch_data(self):
        batch_data = []
        for _ in range(self.batch_size):
            if self.index >= self.n:
                self.index = 0
            batch_data.append(self.shuffle[self.index])
            self.index += 1
        return batch_data
    #
    def __shuffle_data(self):
        random.shuffle(self.data_list)
        return self.data_list
    #
    def __onehot(self, data_list):
        for idx, rank in enumerate(data_list):
            onehot_list = np.zeros(9)
            onehot_list[rank[1] - 1] = 1
            data_list[idx][1] = onehot_list
        #
        return data_list

def split_csv(path, valid_ratio, wanted_cols=[]):
    text_file = pd.read_csv(path, sep='\t', index_col=False, header=None)
    lines = [[] for i in range(len(wanted_cols))]
    for idx, col in enumerate(wanted_cols):     # select column
        lines[idx] = np.array(text_file[col])

    data_list = [[] for i in range(len(text_file))]
    for l in lines:
        for idx, x in enumerate(l):
            data_list[idx].append(x)

    random.shuffle(data_list)
    num_of_valid = math.ceil(len(text_file) * valid_ratio)

    train_data_list = np.array(data_list[num_of_valid:])
    valid_data_list = np.array(data_list[:num_of_valid])

    f = open('splited_train.txt', 'w', newline='')
    wr = csv.writer(f)
    for line in train_data_list:
        wr.writerow(line)
    f.close()

    f = open('splited_valid.txt', 'w', newline='')
    wr = csv.writer(f)
    for line in valid_data_list:
        wr.writerow(line)
    f.close()

    return True
    

def csv2arr(path):
    arr = []
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        temp = []
        for i in line:
            item = 0
            if(i.isnumeric()):
                item = int(i)
            else:
                item = i
            temp.append(item)
        arr.append(temp)
    f.close()

    return arr



class Prediction(object):
    def __init__(self, directory, img_dir, file_dir, num_of_class):
        self.directory = directory
        self.img_dir = img_dir
        self.file_dir = file_dir
        self.num_of_class = num_of_class
        self.og_class = []
        self.onehot_og_class = []
        self.pred_class = []
        self.data_list = self.__read_file()
        self.history = tf.keras.callbacks.CSVLogger(directory+'history/summary.txt', separator="\t", append=False)

    def __read_file(self):
        file_name_list = []
        f = open(self.directory+self.file_dir, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for fname, rank in rdr:
            arr = np.zeros(9)
            temp = int(rank)
            arr[temp-1] = 1
            self.onehot_og_class.append(arr)
            self.og_class.append(temp)
            file_name_list.append(fname)

        d_list = []
        for fname in tqdm(file_name_list): 
            image = np.array(Image.open(self.directory + self.img_dir + fname))
            x = image / 255.
            d_list.append(x)
        d_list = np.array(d_list)
        return d_list

    def prediction(self, model):   
        pred_class = model.predict(self.data_list, 
                                   verbose=1)
        for i in pred_class:
            pred = i.argmax()
            self.pred_class.append(pred + 1)
            
        self.pred_class = np.array(self.pred_class)
        
        count = 0
        for p, o in zip(self.pred_class, self.og_class):
            if(p == o):
                count += 1
        
        return [count, len(self.og_class)], cls_mae(self.onehot_og_class, pred_class)
