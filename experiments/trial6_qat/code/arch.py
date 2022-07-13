''' arch '''
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K


def arch(scale, in_channels, out_channels, channel, blocks):
    ''' arch '''
    inp = Input(shape=(None, None, in_channels))

    x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)
    x = Conv2D(channel, 1, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    upsampled_inp = Lambda(lambda x_list: tf.concat(x_list, axis=3))([inp]*(scale**2))  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

    for i in range(blocks):
        x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Conv2D(channel, 1, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Add()([upsampled_inp, x])

    out = Lambda(lambda x: K.clip(tf.nn.depth_to_space(x, scale), 0., 255.))(x)

    return Model(inputs=inp, outputs=out)


def rep_arch(scale, in_channels, out_channels, channel, blocks):
    ''' rep arch '''
    inp = Input(shape=(None, None, in_channels))

    x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)
    x = Conv2D(channel, 1, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    for i in range(blocks):
        x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Conv2D(channel, 1, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Concatenate()([x, inp])
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros', name='simulation_residual')(x)

    out = Lambda(lambda x: K.clip(tf.nn.depth_to_space(x, scale), 0., 255.))(x)

    return Model(inputs=inp, outputs=out)
