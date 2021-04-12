import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

"""
Unet
Total params: 34,513,345
Trainable params: 34,513,345
Non-trainable params: 0
"""
def getUnet(input_shape=(512, 512, 3)):

    inputs = tf.keras.Input(input_shape)

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

    outputs = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(uconv1_3)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def conv2d(x, filters, kernel_size, padding, activation, kernel_initializer, dropout=0.3):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    return x


""" Nope
Total params: 34,536,897
Trainable params: 34,525,121
Non-trainable params: 11,776
"""
def getUnet_block(input_shape=(512, 512, 3)):

    inputs = tf.keras.Input(input_shape)

    conv1_1 = conv2d(x=inputs, filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    conv1_2 = conv2d(x=conv1_1, filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    
    conv2_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv1_2)
    conv2_2 = conv2d(x=conv2_1, filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    conv2_3 = conv2d(x=conv2_2, filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    
    conv3_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv2_3)
    conv3_2 = conv2d(x=conv3_1, filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    conv3_3 = conv2d(x=conv3_2, filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')

    conv4_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv3_3)
    conv4_2 = conv2d(x=conv4_1, filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    conv4_3 = conv2d(x=conv4_2, filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')

    conv5_1 = MaxPool2D(pool_size=(2,2), strides=2)(conv4_3)
    
    conv5_2 = conv2d(x=conv5_1, filters=1024, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    conv5_3 = conv2d(x=conv5_2, filters=1024, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')

    uconv4 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(conv5_3)
    uconv4_1 = concatenate([uconv4, conv4_3], axis=3)     
    # Crop?
    uconv4_2 = conv2d(x=uconv4_1, filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    uconv4_3 = conv2d(x=uconv4_2, filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
     
    uconv3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv4_3)
    uconv3_1 = concatenate([uconv3, conv3_3], axis=3)
    uconv3_2 = conv2d(x=uconv3_1, filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    uconv3_3 = conv2d(x=uconv3_2, filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
         
    uconv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv3_3)
    uconv2_1 = concatenate([uconv2, conv2_3], axis=3)
    uconv2_2 = conv2d(x=uconv2_1, filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    uconv2_3 = conv2d(x=uconv2_2, filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    
    uconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv2_3)
    uconv1_1 = concatenate([uconv1, conv1_2], axis=3)
    uconv1_2 = conv2d(x=uconv1_1, filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')
    uconv1_3 = conv2d(x=uconv1_2, filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')

    outputs = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(uconv1_3)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


if __name__=="__main__":
    getUnet()
    # getUnet_block()