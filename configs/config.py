import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    IMAGE_PATH = os.getenv("IMAGE_PATH", "answer_sheet.png")

    TOTAL_QUESTIONS = int(os.getenv("TOTAL_QUESTIONS", 50))
    ROWS = int(os.getenv("ROWS", 10))
    COLS = int(os.getenv("COLS", 5))
    OPTIONS = int(os.getenv("OPTIONS", 5))
