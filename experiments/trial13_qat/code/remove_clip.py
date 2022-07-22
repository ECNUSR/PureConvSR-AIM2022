''' remove clip '''
# pylint: disable=no-name-in-module
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow_model_optimization as tfmot
from common.quant import ps_quantization, NoOpQuantizeConfig


def valid_topological_transformation(model1, model2):
    ''' valid topological transformation '''
    input = (np.random.rand(1, 360, 640, 3) * 255).astype(np.int8)
    input_t = tf.constant(input.clip(0, 255))
    out2 = model2(input_t).numpy()
    out1 = model1(input_t).numpy()

    print(abs(out1 - out2).max())
    print(abs(out1 - out2).sum())
    print(np.allclose(out1, out2))


def rep_arch(scale=3, in_channels=3, out_channels=3, channel=28, blocks=3):
    ''' rep arch '''
    inp = Input(shape=(None, None, in_channels))

    x = Conv2D(channel + 3, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for _ in range(blocks):
        x = Conv2D(channel + 3, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    out = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)

    return Model(inputs=inp, outputs=out)

def main():
    ''' main '''
    shutil.copy('experiments/trial13_qat/best_status/', 'experiments/trial13_qat/best_status_clip/')
    model1 = tf.keras.models.load_model('experiments/trial13_qat/best_status_clip/', custom_objects={'tf': tf})
    model2 = rep_arch()
    annotate_model = tf.keras.models.clone_model(model2, clone_function=ps_quantization)
    annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
    with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
        model2 = tfmot.quantization.keras.quantize_apply(annotate_model)

    layers1 = {layer.name: layer for layer in model1.layers}
    layers2 = {layer.name: layer for layer in model2.layers}
    layers2_name = [layer.name for layer in model2.layers]
    for name in layers2_name:
        layer1, layer2 = layers1[name], layers2[name]
        for j in range(len(layer1.weights)):    # pylint: disable=consider-using-enumerate
            layer2.weights[j].assign(layer1.weights[j])
    valid_topological_transformation(model1, model2)
    model1.save('experiments/trial13_qat/best_status_no_clip/', overwrite=True, include_optimizer=True, save_format='tf')
    layers2[layers2_name[-2]].weights[-1].assign(255)
    valid_topological_transformation(model1, model2)
    model1.save('experiments/trial13_qat/best_status/', overwrite=True, include_optimizer=True, save_format='tf')


if __name__ == '__main__':
    main()
