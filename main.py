import random

import torch
import numpy as np

from config import get_config
from runner import MultiWOZRunner

from utils.io_utils import get_or_create_logger

logger = get_or_create_logger(__name__)

def main():
    cfg = get_config()

    # cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

    setattr(cfg, "device", device)
    setattr(cfg, "num_gpus", num_gpus)

    logger.info("Device: {} (the number of GPUs: {})".format(device, num_gpus))

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to {}".format(cfg.seed))

    runner = MultiWOZRunner(cfg)

    if cfg.run_type == "train":
        runner.train()
    else:
        runner.predict()

if __name__ == "__main__":
    main()
