''' arch '''
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K


def arch(scale):
    ''' arch '''
    inp = Input(shape=(None, None, 3))
    x = Conv2D(3*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)
    out = Lambda(lambda x: K.clip(tf.nn.depth_to_space(x, scale), 0., 255.))(x)
    return Model(inputs=inp, outputs=out)
