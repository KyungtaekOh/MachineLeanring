import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def conv2d(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x


def identity(inputs, filters):
    f1, f2, f3 = filters

    x = conv2d(inputs, filters=f1, kernel_size=(1, 1))
    x = conv2d(x, filters=f2, kernel_size=(3, 3))
    x = conv2d(x, filters=f3, kernel_size=(1, 1), activation=None)

    skip = layers.Add()([inputs, x])
    skip = layers.Activation(activation='relu')(skip)

    return skip


def projection(inputs, filters, stride=2):
    f1, f2, f3 = filters

    x = conv2d(inputs, filters=f1, kernel_size=(1, 1), strides=stride)
    x = conv2d(x, filters=f2, kernel_size=(3, 3))
    x = conv2d(x, filters=f3, kernel_size=(1, 1), activation=None)

    proj_x = conv2d(inputs, f3, (1, 1), strides=stride, activation=None)
    skip = layers.Add()([proj_x, x])
    skip = layers.Activation(activation='relu')(skip)

    return skip


"""
Resnet 50
Total params: 23,606,153
Trainable params: 23,553,033
Non-trainable params: 53,120
"""
def getResnet50(num_classes=9, input=(512, 512, 3)):
    inputs = tf.keras.Input(input)
    #
    conv1 = conv2d(inputs, filters=64, kernel_size=(7, 7), strides=2, padding='same')
    #
    conv2_1 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv1)

    filter_2 = [64, 64, 256]
    conv2_2 = projection(conv2_1, filters=filter_2, stride=1)
    conv2_3 = identity(conv2_2, filters=filter_2)
    conv2_4 = identity(conv2_3, filters=filter_2)

    filter_3 = [128, 128, 512]
    conv3_1 = projection(conv2_4, filters=filter_3, stride=2)
    conv3_2 = identity(conv3_1, filters=filter_3)
    conv3_3 = identity(conv3_2, filters=filter_3)
    conv3_4 = identity(conv3_3, filters=filter_3)

    filter_4 = [256, 256, 1024]
    conv4_1 = projection(conv3_4, filters=filter_4, stride=2)
    conv4_2 = identity(conv4_1, filters=filter_4)
    conv4_3 = identity(conv4_2, filters=filter_4)
    conv4_4 = identity(conv4_3, filters=filter_4)
    conv4_5 = identity(conv4_4, filters=filter_4)
    conv4_6 = identity(conv4_5, filters=filter_4)

    filter_5 = [512, 512, 2048]
    conv5_1 = projection(conv4_6, filters=filter_5, stride=2)
    conv5_2 = identity(conv5_1, filters=filter_5)
    conv5_3 = identity(conv5_2, filters=filter_5)

    avg_pooling = layers.GlobalAveragePooling2D()(conv5_3)
    #
    output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(avg_pooling)
    #
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    #
    return model

"""
Resnet 18

Total params: 2,488,073
Trainable params: 2,483,849
Non-trainable params: 4,224
"""

def getResnet(num_classes=9, input=(224, 224, 3)):
    inputs = tf.keras.Input((224, 224, 3))
    #
    conv1 = conv2d(inputs, filters=64, kernel_size=(7, 7), strides=2, padding='same')
    #
    conv1_1 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv1)
    #
    conv2_1 = conv2d(conv1_1, filters=64, kernel_size=(3, 3))
    conv2_2 = conv2d(conv2_1, filters=64, kernel_size=(3, 3), activation=None)
    #
    skip_1 = layers.Add()([conv2_2, conv1_1])
    skip_1 = layers.Activation(activation='relu')(skip_1)
    #
    conv3_1 = conv2d(skip_1, filters=128, kernel_size=(3, 3), strides=2)
    conv3_2 = conv2d(conv3_1, filters=128, kernel_size=(3, 3))
    #
    skip_2 = conv2d(skip_1, filters=128, kernel_size=(1, 1), strides=2, activation=None)
    skip_2 = layers.Add()([conv3_2, skip_2])
    skip_2 = layers.Activation(activation='relu')(skip_2)
    #
    conv4_1 = conv2d(skip_2, filters=256, kernel_size=(3, 3), strides=2)
    conv4_2 = conv2d(conv4_1, filters=256, kernel_size=(3, 3))
    #
    skip_3 = conv2d(skip_2, filters=256, kernel_size=(1, 1), strides=2, activation=None)
    skip_3 = layers.Add()([conv4_2, skip_3])
    skip_3 = layers.Activation(activation='relu')(skip_3)
    #
    conv5_1 = conv2d(skip_3, filters=256, kernel_size=(3, 3), strides=2)
    conv5_2 = conv2d(conv5_1, filters=256, kernel_size=(3, 3))
    #
    skip_4 = conv2d(skip_3, filters=256, kernel_size=(1, 1), strides=2, activation=None)
    skip_4 = layers.Add()([conv5_2, skip_4])
    skip_4 = layers.Activation(activation='relu')(skip_4)
    #
    pool = layers.GlobalAveragePooling2D()(skip_4)
    #
    output = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(pool)
    #
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    #
    return model

from tensorflow.keras.applications.resnet50 import ResNet50
if __name__ == "__main__":
    getResnet50()
    # model = ResNet50(weights = None,classes=9, input_shape=(512, 512, 3))
    # model.summary()
