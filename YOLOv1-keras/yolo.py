import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def conv2d(tensor, filters, kernel_size, strides=1, padding='same', kernel_initialize='he_normal'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initialize)(tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def YOLOv1(input_shape=(448,448,3)):
    inputs = tf.keras.Input(input_shape)

    x = conv2d(inputs, filters=64, kernel_size=(7,7), strides=2)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    x = conv2d(x, filters=192, kernel_size=(3,3))
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)
    
    x = conv2d(x, filters=128, kernel_size=(1,1))
    x = conv2d(x, filters=256, kernel_size=(3,3))
    x = conv2d(x, filters=256, kernel_size=(1,1))
    x = conv2d(x, filters=512, kernel_size=(3,3))
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    for _ in range(4):
        x = conv2d(x, filters=256, kernel_size=(1,1))
        x = conv2d(x, filters=512, kernel_size=(3,3))
    x = conv2d(x, filters=512, kernel_size=(1,1))
    x = conv2d(x, filters=1024, kernel_size=(3,3))   
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)

    for _ in range(2):
        x = conv2d(x, filters=512, kernel_size=(1,1))
        x = conv2d(x, filters=1024, kernel_size=(3,3))
    x = conv2d(x, filters=1024, kernel_size=(3,3))
    x = conv2d(x, filters=1024, kernel_size=(3,3), strides=2)

    x = conv2d(x, filters=1024, kernel_size=(3,3))
    x = conv2d(x, filters=1024, kernel_size=(3,3))
    
    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = Dense(units=4096)(x)
    output = Dense(units=7*7*30, activation='sigmoid')(x) # S*S*(5B + C) - S=7, B=2, C=20
    output = Reshape((7, 7, (20 + 2*5)))(outupt)

    model = Model(inputs=inputs, outputs=output)

    return model    


if __name__=="__main__":
    model = YOLOv1((448,448,3))
    model.summary()