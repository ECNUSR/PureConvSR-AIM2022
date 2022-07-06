''' train '''
import os
import os.path as osp
import argparse
import shutil
import importlib
import tensorflow as tf
from common import logging
from common.lark import try_send
from common.data import DIV2K


def save_gpu_memory():
    ''' save_gpu_memory '''
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', required=True, type=str, help='trial name like id')
    parser.add_argument('--lark', nargs='+', type=str, default=None, help='lark receivers')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default=None)
    args = parser.parse_args()

    # set trail name for save
    config = importlib.import_module(f'trials.{args.trial}.config')
    config.trial_name = args.trial

    # make dirs and init logger
    if not args.resume and osp.exists(osp.join('experiments', args.trial)):
        shutil.rmtree(osp.join('experiments', args.trial))
    if not args.resume and osp.exists(osp.join('tb_logger', args.trial)):
        shutil.rmtree(osp.join('tb_logger', args.trial))
    os.makedirs(osp.join('experiments', args.trial, 'best_status'), exist_ok=True)
    os.makedirs(osp.join('experiments', args.trial, 'visiual'), exist_ok=True)
    if osp.exists(osp.join('experiments', args.trial, 'code')):
        shutil.rmtree(osp.join('experiments', args.trial, 'code'))
    shutil.copytree(f'trials/{args.trial}', osp.join('experiments', args.trial, 'code'))
    os.makedirs(osp.join('tb_logger', args.trial), exist_ok=True)
    logging.init_logger(log_level='INFO',
                        log_file=osp.join('experiments', args.trial, 'info.log'),
                        tb_log_dir=osp.join('tb_logger', args.trial),
                        lark_receiver=args.lark)

    # import trail
    trail = importlib.import_module(f'trials.{args.trial}.solver')

    # create dataset
    train_data = DIV2K(mode='train', scale=3, **config.data)
    logging.info('Create train dataset successfully!')
    logging.info(f'Training: [{len(train_data)}] iterations for each epoch')

    val_data = DIV2K(mode='valid', scale=3)
    logging.info('Create val dataset successfully!')
    logging.info(f'Validating: [{len(val_data)}] iterations for each epoch')

    # create solver
    logging.info(f'Preparing for experiment: [{args.trial}]')
    if args.resume:
        solver = trail.Solver(train_data, val_data, args.resume_path)
    else:
        solver = trail.Solver(train_data, val_data)

    # train
    logging.info('Start training...')
    solver.train()
    try_send('训练完成', 'cjh')


if __name__ == '__main__':
    save_gpu_memory()
    main()