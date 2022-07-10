''' solver '''
# pylint: disable=no-name-in-module
import os.path as osp
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Lambda
import tensorflow_model_optimization as tfmot
from common import logging
from common.quant import NoOpQuantizeConfig, ps_quantization
from common.callbacks import TrainDataShuffleCallback, ValidationCallback


class BaseSolver:
    ''' BaseSolver '''
    def __init__(self, config, arch, train_data, val_data, resume_path=None):
        self.config = config
        self.arch = arch
        self.train_data = train_data
        self.val_data = val_data
        self.resume_path = resume_path
        self.state = {'current_epoch': -1, 'best_epoch': -1, 'best_psnr': -1}

        self.build_model()
        self.build_optimizer()
        self.build_callback()

    def build_model(self):
        ''' build_model '''
        if self.resume_path is not None:
            self.load_resume_model(self.resume_path)
        else:
            self.model = self.arch(**self.config.model)
        logging.info(f'Create model successfully! Params: [{self.model.count_params() / 1e3:.2f}]K')
        self.model.summary(print_fn=logging.info)

    def load_resume_model(self, resume_path):
        ''' load_resume_model '''
        logging.info(f'Load from checkpoint: [{resume_path}]')
        self.model = tf.keras.models.load_model(resume_path, custom_objects={'tf': tf})
        with open(osp.join('experiments', self.config.trial_name, 'state.pkl'), 'rb') as f:
            self.state = pickle.load(f)
            logging.info('Load checkpoint state successfully!')

    def build_optimizer(self):
        ''' build_optimizer '''
        self.optimizer = tf.keras.optimizers.Adam(lr=self.config.train['lr'])

    def build_callback(self):
        ''' build_callback '''
        self.callback = [
            LearningRateScheduler(self.scheduler),
            TrainDataShuffleCallback(self.train_data),
            ValidationCallback(self.config.trial_name, self.val_data, self.state)
        ]

    def train(self):
        ''' train '''
        if self.resume_path is None:
            self.model.compile(optimizer=self.optimizer, loss=self.config.train['loss'])
        self.model.fit(self.train_data, epochs=self.config.train['epochs'], workers=8, callbacks=self.callback, initial_epoch=self.state['current_epoch'] + 1)

    def scheduler(self, epoch):
        ''' scheduler lr '''
        if epoch in self.config.train['lr_steps']:
            current_lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, current_lr * self.config.train['lr_gamma'])
        return K.get_value(self.model.optimizer.lr)


class BaseQuantSolver(BaseSolver):
    ''' BaseQuantSolver '''
    def __init__(self, config, train_data, val_data, resume_path=None, qat_path=None):
        self.qat_path = qat_path
        super().__init__(config, None, train_data, val_data, resume_path)

    def build_model(self):
        if self.resume_path is not None:
            self.load_resume_model(self.resume_path)
        else:
            logging.info('Loading pretrained model ...')
            model = tf.keras.models.load_model(self.qat_path, custom_objects={'tf': tf})
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
