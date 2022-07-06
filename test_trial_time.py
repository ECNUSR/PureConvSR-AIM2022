''' test trial time '''
# pylint: disable=no-name-in-module
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K
import tensorflow_model_optimization as tfmot
from common.quant import NoOpQuantizeConfig, ps_quantization


# set input tensor to [1, 360, 640, 3] for testing time
def representative_dataset_gen_time():
    ''' representative_dataset_gen_time '''
    lr_path = 'datasets/DIV2K/DIV2K_valid_LR_bicubic/X3/0801.pt'
    with open(lr_path, 'rb') as f:
        lr = pickle.load(f)
    lr = lr.astype(np.float32)
    lr = np.expand_dims(lr, 0)
    yield [lr[:, 0:360, 0:640, :]]


def arch(scale=3, in_channels=3, num_fea=32, m=4, out_channels=3):
    ''' arch '''
    inp = Input(shape=(None, None, in_channels))

    x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for i in range(m):
        if i % 2 == 0:
            x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        else:
            x = Conv2D(num_fea, 1, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Concatenate()([x, inp])
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    out = Lambda(lambda x: K.clip(tf.nn.depth_to_space(x, scale), 0., 255.))(x)
    model = Model(inputs=inp, outputs=out)
    return model


def convert_model_quantize(quantized_model_path):
    ''' convert_model_quantize '''
    model = arch()
    model.summary()
    annotate_model = tf.keras.models.clone_model(model, clone_function=ps_quantization)
    annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
    with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
        model = tfmot.quantization.keras.quantize_apply(annotate_model)
    model.save('temp', overwrite=True, include_optimizer=True, save_format='tf')
    model = tf.keras.models.load_model('temp')
    os.system('rm temp -rf')

    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, 360, 640, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter=True
    converter.experimental_new_quantizer=True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen_time
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized_tflite_model = converter.convert()
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_tflite_model)


if __name__ == '__main__':
    convert_model_quantize('test_time.tflite')
