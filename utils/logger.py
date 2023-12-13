import os
import subprocess
import logging
import time
from datetime import timedelta

def init_experiment(params, logger_filename):
    '''
        Initialize the experiment, save parameters and create a logger

        Params:
            - params: a dict contains all hyper-parameters and experimental settings
            - logger_filename: the logger file name
    '''
    # create save path
    get_saved_path(params)

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger


class LogFormatter():
    '''
        A formatter adding date and time informations
    '''
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath):
    '''
        Create logger for the experiment

        Params:
            - filepath: the path which the log file is saved
    '''
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def get_saved_path(params):
    '''
        Create a directory to store the experiment

        Params:
            - params: a dict contains all hyper-parameters and experimental settings
    '''
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.wandb_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    while True:
        run_id = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        dump_path = os.path.join(exp_path, run_id)
        if not os.path.isdir(dump_path):
            break
        else:
            time.sleep(1)
    
    params.dump_path = dump_path
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()

    assert os.path.isdir(params.dump_path)
