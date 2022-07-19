''' solver '''
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from common.solver import BaseSolver, BaseQuantSolver
from common.callbacks import TrainDataShuffleCallback, ValidationWithEMACallback
from . import config, qat_config
from .arch import arch


class TopologicalTransformation(Callback):
    ''' TopologicalTransformation '''
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.iters = 0

    def on_train_begin(self, logs=None):
        ''' on_train_begin '''
        layers2 = tf.keras.models.load_model('experiments/trial11/best_status', custom_objects={'tf': tf}).layers[1]
        weight2, bias2 = layers2.weights[0].numpy(), layers2.weights[1].numpy()
        for layer in self.model.layers:
            if self.name in layer.name:
                weight = layer.weights[0].numpy()
                bias = layer.weights[1].numpy()
                weight[:, :, -3:, :] = weight2
                bias += bias2
                layer.weights[0].assign(weight)
                layer.weights[1].assign(bias)


class Solver(BaseSolver):
    ''' Solver '''
    def __init__(self, train_data, val_data, resume_path=None):
        super().__init__(config, arch, train_data, val_data, resume_path)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            TopologicalTransformation(f'conv2d_{config.model["blocks"] + 1}'),
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]


class SimulationResidual(Callback):
    ''' SimulationResidual '''
    def __init__(self, name):
        super().__init__()
        self.name = name
        layers2 = tf.keras.models.load_model('experiments/trial11_qat/best_status', custom_objects={'tf': tf}).layers[2]
        self.weight = layers2.weights[0].numpy()

    def on_batch_end(self, batch, logs=None):
        ''' on_batch_end '''
        for layer in self.model.layers:
            if self.name in layer.name:
                weight = layer.weights[0].numpy()
                weight[:, :, -3:, :] = self.weight
                layer.weights[0].assign(weight)


class QuantSolver(BaseQuantSolver):
    ''' QuantSolver '''
    def __init__(self, train_data, val_data, resume_path=None, qat_path=None):
        super().__init__(qat_config, train_data, val_data, resume_path, qat_path)

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            SimulationResidual(f'conv2d_{config.model["blocks"] + 1}'),
            ValidationWithEMACallback(self.config.trial_name, self.val_data, self.state)
        ]
