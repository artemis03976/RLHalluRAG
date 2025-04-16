import random
import numpy as np
import torch
import re
import os
from datetime import datetime
import logging


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def clean_text(output):
    return output.split("</think>", 1)[-1].lstrip()


def extract_judgement(output):
    pattern = re.compile(r"\b(Yes|No)\b")
    res = pattern.findall(output)
    if len(res) > 0:
        return res[0].upper()
    return None


def get_logger(args):
    os.makedirs(args.log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(args.log_path, 'train_log_' + timestamp + '.log')
    logging.basicConfig(filename=file_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    return logger