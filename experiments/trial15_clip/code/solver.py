''' solver '''
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Lambda
import tensorflow_model_optimization as tfmot
from common import logging
from common.solver import BaseSolver, BaseQuantSolver
from common.callbacks import TrainDataShuffleCallback, ValidationWithEMACallback
from common.quant import ps_quantization, NoOpQuantizeConfig
from common.remove_clip import remove_clip
from . import config, qat_config, clip_config
from .arch import arch, clip_arch


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

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
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
