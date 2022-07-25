''' logging '''
import os
import logging
from .lark import try_send


initialized_logger = {}


def init_logger(logger_name='ccjiahao', log_level=logging.INFO, log_file=None, tb_log_dir=None, lark_receiver=None):
    ''' init_logger '''
    if logger_name in initialized_logger:
        return initialized_logger[logger_name]
    logger = logging.getLogger(logger_name)
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    if log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    tb_logger = {'root': tb_log_dir} if tb_log_dir is not None else None
    if lark_receiver is not None and isinstance(lark_receiver, str):
        lark_receiver = lark_receiver.split(',')
    initialized_logger[logger_name] = (logger, tb_logger, lark_receiver)


def info(msg, *args, logger_name='ccjiahao', **kwargs):
    ''' logging info '''
    if logger_name not in initialized_logger:
        logging.warning(f'The logger {logger_name} was not initialized before being used.')
    logger, _, _ = init_logger(logger_name)
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, logger_name='ccjiahao', **kwargs):
    ''' logging warning '''
    if logger_name not in initialized_logger:
        logging.warning(f'The logger {logger_name} was not initialized before being used.')
    logger, _, _ = init_logger(logger_name)
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, logger_name='ccjiahao', **kwargs):
    ''' logging error '''
    if logger_name not in initialized_logger:
        logging.warning(f'The logger {logger_name} was not initialized before being used.')
    logger, _, _ = init_logger(logger_name)
    logger.error(msg, *args, **kwargs)


def tb_log(global_step, logger_name='ccjiahao', **kwargs):
    ''' tensorboard log '''
    if logger_name not in initialized_logger:
        logging.warning(f'The logger {logger_name} was not initialized before being used.')
    _, tb_logger, _ = init_logger(logger_name)
    if tb_logger is None:
        return
    for key, value in kwargs.items():
        keys = key.split('//')
        key, path = keys[-1], os.path.join(tb_logger['root'], *keys[:-1])
        if path not in tb_logger:
            from tensorboardX import SummaryWriter
            tb_logger[path] = SummaryWriter(log_dir=path)
        logger = tb_logger[path]
        try:
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if isinstance(value, (float, int)):
                logger.add_scalar(key, value, global_step)
            elif isinstance(value, str):
                logger.add_text(key, value, global_step)
        except Exception:
            pass


def report(msg: str, logger_name='ccjiahao'):
    ''' lark report '''
    if logger_name not in initialized_logger:
        logging.warning(f'The logger {logger_name} was not initialized before being used.')
    _, _, whos = init_logger(logger_name)
    if whos is None:
        return
    if not isinstance(whos, (list, tuple)):
        whos = [whos]
    try_send(msg, *whos)
