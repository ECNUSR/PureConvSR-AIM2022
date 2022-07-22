''' train '''
import os
import os.path as osp
import argparse
import shutil
import importlib
import tensorflow as tf
from common import logging
from common.data import DIV2K


def save_gpu_memory():
    ''' save_gpu_memory '''
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', required=True, type=str, help='trial name like id')
    parser.add_argument('--df2k', action='store_true', default=False)
    parser.add_argument('--qat_path', default=None, type=str, help='qat path')
    parser.add_argument('--lark', nargs='+', type=str, default=None, help='lark receivers')
    args = parser.parse_args()

    # set trail name for save
    config = importlib.import_module(f'trials.{args.trial}.clip_config')
    config.trial_name = f'{args.trial}_clip'

    # make dirs and init logger
    if osp.exists(osp.join('experiments', config.trial_name)):
        shutil.rmtree(osp.join('experiments', config.trial_name))
    if osp.exists(osp.join('tb_logger', config.trial_name)):
        shutil.rmtree(osp.join('tb_logger', config.trial_name))
    os.makedirs(osp.join('experiments', config.trial_name, 'best_status'), exist_ok=True)
    os.makedirs(osp.join('experiments', config.trial_name, 'visiual'), exist_ok=True)
    if osp.exists(osp.join('experiments', config.trial_name, 'code')):
        shutil.rmtree(osp.join('experiments', config.trial_name, 'code'))
    shutil.copytree(f'trials/{args.trial}', osp.join('experiments', config.trial_name, 'code'))
    os.makedirs(osp.join('tb_logger', config.trial_name), exist_ok=True)
    logging.init_logger(log_level='INFO',
                        log_file=osp.join('experiments', config.trial_name, 'info.log'),
                        tb_log_dir=osp.join('tb_logger', config.trial_name),
                        lark_receiver=args.lark)

    # import trail
    trail = importlib.import_module(f'trials.{args.trial}.solver')

    # create dataset
    if args.df2k:
        train_data = DIV2K(mode='df2k', scale=3, **config.data)
    else:
        train_data = DIV2K(mode='train', scale=3, **config.data)
    logging.info('Create train dataset successfully!')
    logging.info(f'Training: [{len(train_data)}] iterations for each epoch')

    val_data = DIV2K(mode='valid', scale=3)
    logging.info('Create val dataset successfully!')
    logging.info(f'Validating: [{len(val_data)}] iterations for each epoch')

    # create solver
    logging.info(f'Preparing for experiment: [{config.trial_name}]')
    solver = trail.RemoveClipQuantSolver(train_data, val_data, qat_path=args.qat_path)

    # train
    logging.info('Start training...')
    solver.train()
    logging.report('训练完成')


if __name__ == '__main__':
    save_gpu_memory()
    main()
