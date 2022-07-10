''' solver '''
# pylint: disable=no-name-in-module
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import Callback
import tensorflow_model_optimization as tfmot
from common import logging
from common.solver import BaseQuantSolver
from common.callbacks import TrainDataShuffleCallback, ValidationCallback
from common.quant import NoOpQuantizeConfig, ps_quantization
from . import qat_config
from .arch import rep_arch


class SimulationResidual(Callback):
    ''' SimulationResidual '''
    def on_batch_end(self, batch, logs=None):
        ''' on_batch_end '''
        for layer in self.model.layers:
            if 'simulation_residual' in layer.name:
                weight = layer.weights[0].numpy()
                channel = weight.shape[2] - 3
                weight2 = np.zeros((3, 3, channel + 3, 27))
                weight2[:, :, :channel, :] = weight[:, :, :channel, :]
                weight2[:, :, channel:, :] = 9.9994359e-04
                for j in range(27):
                    weight2[1, 1, channel + j % 3, j] = 1.0009998e+00
                layer.weights[0].assign(weight2)


class QuantSolver(BaseQuantSolver):
    ''' QuantSolver '''
    def __init__(self, train_data, val_data, resume_path=None, qat_path=None):
        super().__init__(qat_config, train_data, val_data, resume_path, qat_path)

    def topological_transformation(self, qat_path):
        ''' topological transformation '''
        model1 = tf.keras.models.load_model(qat_path, custom_objects={'tf': tf})
        model2 = rep_arch(**self.config.model)
        goal_step = self.config.model['blocks'] + 1
        for i in range(goal_step):
            name = 'conv2d' if i == 0 else f'conv2d_{i}'
            for layer in model1.layers:
                if layer.name == name:
                    layers1 = layer
            for layer in model2.layers:
                if layer.name == name:
                    layers2 = layer
            layers2.weights[0].assign(layers1.weights[0].numpy())
            layers2.weights[1].assign(layers1.weights[1].numpy())
        for layer in model1.layers:
            if layer.name == f'conv2d_{goal_step}':
                layers1 = layer
        for layer in model2.layers:
            if layer.name == 'simulation_residual':
                layers2 = layer
        weight = np.zeros((3, 3, self.config.model['channel'] + 3, 27))
        weight[:, :, :self.config.model['channel'], :] = layers1.weights[0].numpy()
        bias = layers1.weights[1].numpy()
        for j in range(27):
            weight[1, 1, self.config.model['channel'] + j % 3, j] = 1
        layers2.weights[0].assign(weight)
        layers2.weights[1].assign(bias)
        self.valid_topological_transformation(model1, model2)
        return model2

    @staticmethod
    def valid_topological_transformation(model1, model2):
        ''' valid topological transformation '''
        input = (np.random.rand(1, 360, 640, 3) * 255).astype(np.int8)
        input_t = tf.constant(input)
        out1 = model1(input_t).numpy()
        out2 = model2(input_t).numpy()

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

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            SimulationResidual(),
            ValidationCallback(self.config.trial_name, self.val_data, self.state)
        ]
