''' arch '''
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K


def rep_arch(scale=3, in_channels=3, out_channels=3, channel=28, blocks=4):
    ''' rep_arch '''
    inp = Input(shape=(None, None, in_channels))

    # Feature extraction
    x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for _ in range(blocks):
        x = Conv2D(channel, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Concatenate()([x, inp])
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros', name='simulation_residual')(x)

    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out)

    return Model(inputs=inp, outputs=out)
