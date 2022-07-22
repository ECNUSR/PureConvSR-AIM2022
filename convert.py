''' convert '''
import pickle
import math
import os
import os.path as osp
import argparse
import numpy as np
import tqdm
import cv2
import tensorflow as tf


def save_gpu_memory():
    ''' save_gpu_memory '''
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)


def representative_dataset_gen():
    ''' representative_dataset_gen '''
    for i in tqdm.tqdm(range(801, 901), desc='rep_dataset'):
        lr_path = f'datasets/DIV2K/DIV2K_valid_LR_bicubic/X3/{i:04d}.pt'
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        lr = lr.astype(np.float32)
        lr = np.expand_dims(lr, 0)
        yield [lr]


# set input tensor to [1, 360, 640, 3] for testing time
def representative_dataset_gen_time():
    ''' representative_dataset_gen_time '''
    lr_path = 'datasets/DIV2K/DIV2K_valid_LR_bicubic/X3/0801.pt'
    with open(lr_path, 'rb') as f:
        lr = pickle.load(f)
    lr = lr.astype(np.float32)
    lr = np.expand_dims(lr, 0)
    yield [lr[:, 0:360, 0:640, :]]


def convert_model(model_path, tflite_path):
    ''' convert_model '''
    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, None, None, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter=True
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


def convert_model_quantize(model_path, tflite_path, time=False):
    ''' convert_model_quantize '''
    if time:
        tensor_shape = [1, 360, 640, 3]
        rep = representative_dataset_gen_time
    else:
        tensor_shape = [1, None, None, 3]
        rep = representative_dataset_gen

    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(tensor_shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter=True
    converter.experimental_new_quantizer=True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized_tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(quantized_tflite_model)


def evaluate(tflite_path, save_path):
    ''' evaluate '''
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size, input_zero_point = input_details[0]['quantization']
    output_size, output_zero_point = output_details[0]['quantization']
    print(f'Input Scale: {input_size}, Zero Point: {input_zero_point}')
    print(f'Output Scale: {output_size}, Zero Point: {output_zero_point}')

    psnr = 0.0
    psnr_bar = tqdm.tqdm(range(801, 901), desc='calc_psnr')
    for i in psnr_bar:
        with open(f'datasets/DIV2K/DIV2K_valid_LR_bicubic/X3/0{i}.pt', 'rb') as f:
            lr = pickle.load(f)
        lr = np.expand_dims(lr, 0).astype(np.float32)
        # lr = np.round(lr / input_size + input_zero_point).astype(np.uint8)
        lr = lr.astype(np.uint8)
        interpreter.resize_tensor_input(input_details[0]['index'], lr.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], lr)
        interpreter.invoke()

        sr = interpreter.get_tensor(output_details[0]['index'])
        # sr = np.clip(np.round((sr.astype(np.float32) - output_zero_point) * output_size), 0, 255)
        sr = np.clip(sr, 0, 255)

        # save image
        cv2.imwrite(osp.join(save_path, f'{i:04d}.png'), cv2.cvtColor(sr.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR))

        # calc psnr
        with open(f'datasets/DIV2K/DIV2K_valid_HR/0{i}.pt', 'rb') as f:
            hr = pickle.load(f)
        hr = np.expand_dims(hr, 0).astype(np.float32)
        mse = np.mean((sr.astype(np.float32) - hr.astype(np.float32)) ** 2)
        singlepsnr =  20. * math.log10(255. / math.sqrt(mse))
        psnr += singlepsnr
        psnr_bar.set_description(f'psnr: {psnr / (i - 800)}')
    print(psnr / 100)


def main():
    ''' main '''
    # save gpu memory
    save_gpu_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='trial name')
    parser.add_argument('--clip', action='store_true', default=False)
    args = parser.parse_args()

    # convert model
    os.makedirs(f'TFLite/{args.name}/', exist_ok=True)
    if args.clip:
        ## model.tflite (INT8, 360)
        convert_model_quantize(f'experiments/{args.name}_clip/best_status', f'TFLite/{args.name}/model.tflite', time=True)
        ## model_none.tflite (INT8, None)
        convert_model_quantize(f'experiments/{args.name}_clip/best_status', f'TFLite/{args.name}/model_none.tflite', time=False)
    else:
        ## model.tflite (INT8, 360)
        convert_model_quantize(f'experiments/{args.name}_qat/best_status', f'TFLite/{args.name}/model.tflite', time=True)
        ## model_none.tflite (INT8, None)
        convert_model_quantize(f'experiments/{args.name}_qat/best_status', f'TFLite/{args.name}/model_none.tflite', time=False)
    ## model_none_float.tflite (float32, None)
    convert_model(f'experiments/{args.name}/best_status', f'TFLite/{args.name}/model_none_float.tflite')

    # evaluate with model_none.tflite
    evaluate(f'TFLite/{args.name}/model_none.tflite', f'experiments/{args.name}_qat/visiual')


if __name__ == '__main__':
    main()
