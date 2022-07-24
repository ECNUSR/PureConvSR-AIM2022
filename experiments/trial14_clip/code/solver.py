''' solver '''
# pylint: disable=no-name-in-module
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import Callback
import tensorflow_model_optimization as tfmot
from common import logging
from common.solver import BaseSolver, BaseQuantSolver
from common.callbacks import TrainDataShuffleCallback, ValidationWithEMACallback
from common.quant import ps_quantization, NoOpQuantizeConfig
from common.remove_clip import remove_clip
from . import config, qat_config, clip_config
from .arch import arch, rep_arch, clip_arch


class SimulationResidual(Callback):
    ''' SimulationResidual '''
    def __init__(self, goal_step):
        super().__init__()
        self.goal_step = goal_step

    def on_batch_end(self, batch, logs=None):
        goal_step = self.goal_step
        for i in range(goal_step + 1):
            name = 'conv2d' if i == 0 else f'conv2d_{i}'
            for layer_ in self.model.layers:
                if name in layer_.name:
                    layer = layer_
                    break
            weight, bias = layer.weights[0].numpy(), layer.weights[1].numpy()
            kernel_size, channel = weight.shape[0], weight.shape[2] - 3
            if i == goal_step:
                weight[:, :, -3:, :] = 0
                for j in range(27):
                    weight[kernel_size//2, kernel_size//2, channel + j % 3, j] = 1
            else:
                if i != 0:
                    weight[:, :, -3:, :] = 0
                weight[:, :, :, -3:] = 0
                for j in [1, 2, 3]:
                    weight[kernel_size//2, kernel_size//2, -j, -j] = 1
                bias[-3:] = 0
            layer.weights[0].assign(weight)
            layer.weights[1].assign(bias)


class Solver(BaseSolver):
    ''' Solver '''
    def __init__(self, train_data, val_data, resume_path=None):
        super().__init__(config, arch, train_data, val_data, resume_path)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]


class QuantSolver(BaseQuantSolver):
    ''' QuantSolver '''
    def __init__(self, train_data, val_data, resume_path=None, qat_path=None):
        super().__init__(qat_config, train_data, val_data, resume_path, qat_path)

    def topological_transformation(self, qat_path):
        ''' topological transformation '''
        model1 = tf.keras.models.load_model(qat_path, custom_objects={'tf': tf})
        model2 = rep_arch(**config.model)
        goal_step = config.model['blocks'] + 1
        for i in range(goal_step + 1):
            name = 'conv2d' if i == 0 else f'conv2d_{i}'
            for layer in model1.layers:
                if layer.name == name:
                    layers1 = layer
            for layer in model2.layers:
                if layer.name == name:
                    layers2 = layer
            if i == goal_step:
                weight = np.zeros_like(layers2.weights[0].numpy())
                bias = layers1.weights[1].numpy()
                weight[:, :, :-3, :] = layers1.weights[0].numpy()
                k, c = layers1.weights[0].shape[1:3]
                for j in range(27):
                    weight[k//2, k//2, c + j % 3, j] = 1
            elif i == 0:
                weight = np.zeros_like(layers2.weights[0].numpy())
                bias = np.zeros_like(layers2.weights[1].numpy())
                weight[:, :, :, :-3] = layers1.weights[0].numpy()
                k, c = layers1.weights[0].shape[1:3]
                for j in [-1, -2, -3]:
                    weight[k//2, k//2, j, j] = 1
                bias[:-3] = layers1.weights[1].numpy()
            else:
                weight = np.zeros_like(layers2.weights[0].numpy())
                bias = np.zeros_like(layers2.weights[1].numpy())
                weight[:, :, :-3, :-3] = layers1.weights[0].numpy()
                k, c = layers1.weights[0].shape[1:3]
                for j in [-1, -2, -3]:
                    weight[k//2, k//2, j, j] = 1
                bias[:-3] = layers1.weights[1].numpy()
            layers2.weights[0].assign(weight)
            layers2.weights[1].assign(bias)
        self.valid_topological_transformation(model1, model2)
        return model2

    @staticmethod
    def valid_topological_transformation(model1, model2):
        ''' valid topological transformation '''
        input = (np.random.rand(1, 360, 640, 3) * 255).astype(np.int8)
        input_t = tf.constant(input.clip(0, 255))
        out1 = model1(input_t).numpy().clip(0, 255)
        out2 = model2(input_t).numpy().clip(0, 255)

        print(abs(out1 - out2).max())
        print(abs(out1 - out2).sum())
        print(np.allclose(out1, out2))
        assert abs(out1 - out2).max() < 1e-3

    def build_model(self):
        ''' build model '''
        if self.resume_path is not None:
            self.load_resume_model(self.resume_path)
        else:
            logging.info('Loading pretrained model ...')
            model = self.topological_transformation(self.qat_path)
            logging.info('Start copying weights and annotate Lambda layer...')
            annotate_model = tf.keras.models.clone_model(model, clone_function=ps_quantization)
            logging.info('Start annotating other parts of model...')
            annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
            logging.info('Creating quantize-aware model...')
            depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, self.config.scale))
            with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
                self.model = tfmot.quantization.keras.quantize_apply(annotate_model)
        logging.info(f'Create model successfully! Params: [{self.model.count_params() / 1e3:.2f}]K')
        self.model.summary(print_fn=logging.info)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            SimulationResidual(config.model['blocks'] + 1),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]


class RemoveClipQuantSolver(BaseQuantSolver):
    ''' RemoveClipQuantSolver '''
    def __init__(self, train_data, val_data, resume_path=None, qat_path=None):
        super().__init__(clip_config, train_data, val_data, resume_path, qat_path)

    def build_model(self):
        ''' build model '''
        model1 = tf.keras.models.load_model(self.qat_path, custom_objects={'tf': tf})
        model2 = clip_arch(**clip_config.model)
        annotate_model = tf.keras.models.clone_model(model2, clone_function=ps_quantization)
        annotate_model = tfmot.quantization.keras.quantize_annotate_model(annotate_model)
        depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, 3))
        with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig, 'depth_to_space': depth_to_space, 'tf': tf}):
            model2 = tfmot.quantization.keras.quantize_apply(annotate_model)
        self.model = remove_clip(model1, model2)
        self.model.summary(print_fn=logging.info)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]
