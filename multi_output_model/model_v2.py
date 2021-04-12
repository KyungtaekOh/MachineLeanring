import numpy as np
import tensorflow as tf
import model as model_v1
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class crop_layer(keras.layers.Layer):
    def __init__(self, name=None, batch_size=None):
        super(crop_layer, self).__init__(name = name)
        self.batch_size = batch_size

    def call(self, image, tensor, threshold_value=0.7):
        
        inputs = tf.cast(tf.where(tensor > threshold_value, 1, 0), np.int32)
        b, h, w, d = inputs.shape

        crop_img = []
        for batch in range(self.batch_size):
            mask = inputs[batch,:,:,0]
            # IndexError

            sum_of_row = tf.reduce_sum(mask[:,:], axis=0)
            sum_of_col = tf.reduce_sum(mask[:,:], axis=1)

            # tf.errors.InvalidArgumentError
            left = tf.where(sum_of_col != 0)[0][0]
            right = tf.where(sum_of_col != 0)[-1][0]
            top = tf.where(sum_of_row != 0)[0][0]
            bot = tf.where(sum_of_row != 0)[-1][0]
                
            img = image[batch, left:right, top:bot, :]

            img = tf.image.resize(img, [400, 400])
            crop_img.append(img)

        crop_img = tf.convert_to_tensor(crop_img)
        return crop_img

def getModel_v2(inputs_shape=(512, 512, 3), num_classes=9, batch_size=4, summary=True):

    inputs = tf.keras.Input(inputs_shape, name='s_in')

    ################
    # Segmentation #
    ################
    conv1_1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    conv1_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv1_1)
    
    conv2_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv1_2)
    conv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv2_1)
    conv2_3 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv2_2)
    
    conv3_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv2_3)
    conv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv3_2)

    conv4_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv3_3)
    conv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv4_2)

    conv5_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv4_3)
    
    conv5_2 = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv5_1)
    conv5_3 = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv5_2)

    uconv4 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(conv5_3)
    uconv4_1 = concatenate([uconv4, conv4_3], axis=3)
    uconv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv4_1)
    uconv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv4_2)
     
    uconv3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv4_3)
    uconv3_1 = concatenate([uconv3, conv3_3], axis=3)
    uconv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv3_1)
    uconv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv3_2)
         
    uconv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv3_3)
    uconv2_1 = concatenate([uconv2, conv2_3], axis=3)
    uconv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv2_1)
    uconv2_3 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv2_2)
    
    uconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv2_3)
    uconv1_1 = concatenate([uconv1, conv1_2], axis=3)
    uconv1_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv1_1)
    uconv1_3 = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(uconv1_2)

    seg_out = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', name='s_out', kernel_initializer='he_normal')(uconv1_3)

    #########################
    # segmentation and crop #
    #########################
    crop = crop_layer(name = 'crop_layer', batch_size=batch_size)
    
    # if tf.reduce_max(seg_out) > 0:
    #     input2 = crop(inputs, seg_out[:,:,:,:])
    input2 = crop(inputs, seg_out[:,:,:,:])
    ##############################
    # Classification             #
    # Input_size : (400, 400, 3) #
    ##############################
    modules = model_v1.Inception_resnet_v2()

    x = modules.stem(input2)
    for i in range(modules.param['part_a']):
        x = modules.inception_a(x)
    
    x = modules.reduction_a(x)
    for i in range(modules.param['part_b']):
        x = modules.inception_b(x)

    x = modules.reduction_b(x)
    for i in range(modules.param['part_c']):
        x = modules.inception_c(x)

    x = GlobalAveragePooling2D()(x)


    class_out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='c_out')(x)

    ### Model ###
    model = Model(inputs=inputs, outputs=[seg_out, class_out])
    if summary:
        model.summary()

    return model


if __name__=="__main__":
    getModel_v2()
