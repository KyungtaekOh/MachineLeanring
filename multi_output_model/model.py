import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

"""
Inception-Resnet-v2
Total params: 22,551,729
Trainable params: 22,489,441
Non-trainable params: 62,288

Unet
Total params: 34,513,345
Trainable params: 34,513,345
Non-trainable params: 0

total
Total params: 57,065,074
Trainable params: 57,002,786
Non-trainable params: 62,288
"""
def getModel(input_shape=(512,512,3), num_classes=9):
    inputs = tf.keras.Input(input_shape)

    ########
    # Unet #
    ########
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


    #######################
    # Inception-Resnet-v2 #
    #######################
    modules = Inception_resnet_v2()

    x = modules.stem(inputs)
    for i in range(modules.param['part_a']):
        x = modules.inception_a(x)
    
    x = modules.reduction_a(x)
    for i in range(modules.param['part_b']):
        x = modules.inception_b(x)

    x = modules.reduction_b(x)
    for i in range(modules.param['part_c']):
        x = modules.inception_c(x)

    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.8)(x)


    seg_out = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid', name='s_out')(uconv1_3)
    class_out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='c_out')(x)

    ### Model ###
    model = Model(inputs=inputs, outputs=[seg_out, class_out])
    model.summary()

    return model



class Inception_resnet_v2:

    def __init__(self):
        # k    l    m    n
        # 256  256  384  384
        self.param = {'k':256,
                      'l':256,
                      'm':384,
                      'n':384,
                      'part_a':4,
                      'part_b':8,
                      'part_c':4}

    def conv2d(self, x, filters, kernel_size, strides=1, padding='same', activation='relu'):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x

    def stem(self, inputs):
        x = self.conv2d(inputs, filters=32, kernel_size=(3,3), strides=2, padding='valid')
        x = self.conv2d(x, filters=32, kernel_size=(3,3), padding='valid')
        x = self.conv2d(x, filters=64, kernel_size=(3,3), padding='valid')

        x_l = MaxPool2D(pool_size=(3,3), strides=2, padding='valid')(x)
        x_r = self.conv2d(x, filters=96, kernel_size=(3,3), strides=2, padding='valid')

        x = concatenate([x_l, x_r])

        x_l = self.conv2d(x, filters=64, kernel_size=(1,1))
        x_l = self.conv2d(x_l, filters=96, kernel_size=(3,3), padding='valid')

        x_r = self.conv2d(x, filters=64, kernel_size=(1,1))
        x_r = self.conv2d(x_r, filters=64, kernel_size=(1,7))
        x_r = self.conv2d(x_r, filters=64, kernel_size=(7,1))
        x_r = self.conv2d(x_r, filters=96, kernel_size=(3,3), padding='valid')

        x = concatenate([x_l, x_r])

        x_l = self.conv2d(x, filters=192, kernel_size=(3,3), strides=1, padding='valid')
        x_r = MaxPool2D(pool_size=(3,3), strides=1, padding='valid')(x)

        out = concatenate([x_l, x_r])   # (None, 121, 121, 384)

        return out

    def inception_a(self, inputs):
        
        conv1 = self.conv2d(inputs, filters=32, kernel_size=(1,1))
        conv2 = self.conv2d(conv1, filters=32, kernel_size=(3,3))
        conv3 = self.conv2d(conv1, filters=48, kernel_size=(3,3))
        conv3 = self.conv2d(conv3, filters=64, kernel_size=(3,3))

        x = concatenate([conv1, conv2, conv3])
        x = self.conv2d(x, filters=384, kernel_size=(1,1))

        out = Add()([inputs, x])
        out = Activation(activation='relu')(out)

        return out

    def inception_b(self, inputs):

        conv1 = self.conv2d(inputs, filters=192, kernel_size=(1,1))

        conv2_1 = self.conv2d(inputs, filters=128, kernel_size=(1,1))
        conv2_2 = self.conv2d(conv2_1, filters=160, kernel_size=(1,7))
        conv2_3 = self.conv2d(conv2_2, filters=192, kernel_size=(7,1))

        concat = concatenate([conv1, conv2_3])
        conv_c = self.conv2d(concat, filters=1152, kernel_size=(1,1))

        out = Add()([inputs, conv_c])
        out = Activation('relu')(out)

        return out

    def inception_c(self, inputs):

        conv1 = self.conv2d(inputs, filters=192, kernel_size=(1,1))

        conv2 = self.conv2d(conv1, filters=224, kernel_size=(1,3))
        conv2 = self.conv2d(conv2, filters=226, kernel_size=(3,1))
        
        concat = concatenate([conv1, conv2])
        conv_c = self.conv2d(concat, filters=2048, kernel_size=(1,1))

        out = Add()([inputs, conv_c])
        out = Activation("relu")(out)

        return out

    def reduction_a(self, inputs):
        pool = MaxPool2D(pool_size=(3,3), strides=2, padding='valid')(inputs)
        
        conv1 = self.conv2d(inputs, filters=self.param['n'], kernel_size=(3,3), strides=2, padding='valid')

        conv2 = self.conv2d(inputs, filters=self.param['k'], kernel_size=(1,1))
        conv2 = self.conv2d(conv2, filters=self.param['l'], kernel_size=(3,3))
        conv2 = self.conv2d(conv2, filters=self.param['m'], kernel_size=(3,3), strides=2, padding='valid')

        out = concatenate([pool, conv1, conv2])

        return out

    def reduction_b(self, inputs):
        pool = MaxPool2D(pool_size=(3,3), strides=2, padding='valid')(inputs)

        conv = self.conv2d(inputs, filters=256, kernel_size=(1,1))

        conv1 = self.conv2d(conv, filters=384, kernel_size=(3,3), strides=2, padding='valid')

        conv2 = self.conv2d(conv, filters=256, kernel_size=(3,3), strides=2, padding='valid')

        conv3 = self.conv2d(conv, filters=256, kernel_size=(3,3))
        conv3 = self.conv2d(conv3, filters=256, kernel_size=(3,3), strides=2, padding='valid')

        out = concatenate([pool, conv1, conv2, conv3])

        return out

if __name__=="__main__":
    getModel()