import logging
import numpy as np
import random
import torch


def creat_log(loggerName: str, logPath: str):
    """Create logger for training process"""
    
    # Create logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(level=logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(logPath)
    file_handler.setLevel(level=logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Create formatter
    fmt = "[%(asctime)s] %(filename)s - %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # Add handler and formatter to logger
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
