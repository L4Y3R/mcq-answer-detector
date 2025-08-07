import logging
import os
from configs.config import Config

def setup_logger(name="MCQAnswerDetector"):
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL.upper())

    log_path = os.path.join(Config.LOG_DIR, "mcq_detector.log")
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger
