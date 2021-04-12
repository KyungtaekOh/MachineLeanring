from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import shutil
import math
import csv


class DataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, data_list, batch_size, target_size=(512, 512), shuffle=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.n = len(data_list)

        if self.shuffle:
            random.shuffle(self.data_list)
#
    def __len__(self):
        return math.ceil(len(self.data_list)/self.batch_size)
#
    def __getitem__(self, index):
        data_batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        xs, ys = self.__data_generation(data_batch)

        return xs, ys
#
    def __data_generation(self, data_list_temp):
        x, y = [[] for i in range(len(data_list_temp))], [[] for i in range(len(data_list_temp))]
        for idx, (img, msk) in enumerate(data_list_temp):
            image = np.array(Image.open(self.img_dir +'/'+ img))
            mask = np.array(Image.open(self.mask_dir +'/'+ msk))
            mask = mask.reshape(mask.shape+(1,))    # (512, 512, 1)
            x[idx] = image / 255.                   # (512, 512, 3)
            y[idx] = mask // 255
        x = np.array(x)         # (16, 512, 512, 3) 
        y = np.array(y)         # (16, 512, 512, 1)
        
        return x, y


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


def csv2arr(file_path, desired_cols=[0, 1]):
    arr = []
    f = open(file_path, 'r', encoding = 'cp949')
    rdr = csv.reader(f, delimiter='\t')
    for line in rdr:
        temp = []
        for i in desired_cols: 
            temp.append(line[i])
        arr.append(temp)

    return arr


class Prediction(object):
    def __init__(self, data_list, img_dir, mask_dir, output_dir, model):
        self.data_list = data_list  # arr(tools.csv2arr) from main.py
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.img_list, self.mask_list = self.__read_csv(data_list)
        self.model = model          # after load_weights()
        self.iou = 0
        self.n = len(self.data_list)
        self.threshold_value = 0.7
        self.pred = self.__prediction(self.model)

    def __prediction(self, model):
        pred_list = []
        sum_of_iou = 0
        pred = model.predict(self.img_list, verbose=1)
        for idx, (img_name, mask_name) in tqdm(enumerate(self.data_list)):
            pre = pred[idx].reshape(self.mask_list[idx].shape[0], self.mask_list[idx].shape[1])
            pre = np.where(pre > self.threshold_value, 255, 0).astype(np.uint8)
            pred_list.append(pre)
            
            ### Save Image and Prediction
            # if not (self.__save_pred(pre, img_name)):
            #     print("Something wrong with save process!")
            sum_of_iou += self.__cls_iou(self.mask_list[idx], pre/255)
        self.iou = sum_of_iou/self.n
        return np.array(pred_list)

    def __cls_iou(self, true, pred):
        intersection = tf.reduce_sum((true*pred).astype(np.int32))
        union = tf.reduce_sum(np.where(true+pred > 0, 1, 0).astype(np.int32))

        return intersection / union

    def __save_pred(self, pred, file_name):
        img = Image.fromarray(pred)
        fname = file_name.split('.')
        img.save(self.output_dir + fname[0]+'_PRED.'+fname[1])

        shutil.copyfile(self.img_dir+file_name, self.output_dir+file_name)

        return True

    def __read_csv(self, name_list):
        img_list, mask_list = [], []
        print("Start read csv file.")
        for img_name, mask_name in tqdm(name_list):
            img = np.array(Image.open(self.img_dir+img_name))/255.
            img_list.append(img)
            mask = np.array(Image.open(self.mask_dir+mask_name))/255.
            mask_list.append(mask)
        
        return np.array(img_list), np.array(mask_list)



"""
def cls_IoU(true, pred):
    true = np.array(true).reshape(true.shape[0], true.shape[1])
    pred = np.array(pred).reshape(pred.shape[0], pred.shape[1])
    
    if(true.max() == 255):
        true=true//255
    if(pred.max() == 255):
        pred=pred//255

    intersection = tf.reduce_sum((true*pred).astype(np.int32))
    union = tf.reduce_sum(np.where(true+pred > 0, 1, 0).astype(np.int32))
    
    return intersection / union
"""


"""     
        # self.argument = dict(rotation_range = 0.2,
        #                      width_shift_range = 0.1,
        #                      height_shift_range = 0.1,
        #                      shear_range = 0.1,
        #                      zoom_range = 0.1,
        #                      horizontal_flip = True,
        #                      vertical_flip=False,
        #                      fill_mode = 'nearest'))


        imageDataGene = ImageDataGenerator(**self.argument)
        maskDataGene = ImageDataGenerator(**self.argument)
        imageGene = imageDataGene.flow(x,
                                       batch_size=self.batch_size,
                                       seed=1)
        maskGene = maskDataGene.flow(y,
                                     batch_size=self.batch_size,
                                     seed=1)
        return imageGene, maskGene
"""
