import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def create_exp_dir(logs_dir: Path, exp_name: str):
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d_%H-%M")
    exp_dir = logs_dir / exp_name / date_string
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
