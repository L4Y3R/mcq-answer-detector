import os
import cv2
import numpy as np
from configs.config import Config
from exceptions.exceptions import (
    ImageNotFoundError,
    ImageProcessingError,
    GridDetectionError,
    AnswerExtractionError,
    MCQDetectorError
)

class MCQAnswerDetector:
    def __init__(self, logger):
        self.logger = logger
        self.total_questions = Config.TOTAL_QUESTIONS
        self.rows = Config.ROWS
        self.cols = Config.COLS
        self.options = Config.OPTIONS

    def preprocess_image(self, image_path):
        try:
            if not image_path or not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                raise ImageNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                self.logger.error("Failed to read image.")
                raise ImageProcessingError("Could not decode image.")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            debug_path = f"{Config.LOG_DIR}/debug_blurred.png"
            cv2.imwrite(debug_path, blurred)
            self.logger.info(f"Blurred image saved at {debug_path}.")

            return image, gray, blurred

        except Exception as e:
            self.logger.exception("Error during image preprocessing.")
            raise ImageProcessingError(str(e))

    def get_grid_region(self, gray_image):
        try:
            h, w = gray_image.shape
            top, bottom = int(h * 0.25), int(h * 0.95)
            left, right = int(w * 0.06), int(w * 0.985)
            cropped = gray_image[top:bottom, left:right]

            self.logger.info(f"Cropped grid: top={top}, bottom={bottom}, left={left}, right={right}")
            return cropped

        except Exception as e:
            self.logger.exception("Error cropping the grid region.")
            raise GridDetectionError("Grid region could not be determined.")

    def extract_answers(self, cropped_img):
        try:
            answers = []
            h, w = cropped_img.shape
            cell_h = h // self.rows
            col_w = w // self.cols
            opt_w = col_w // self.options

            if cell_h == 0 or col_w == 0 or opt_w == 0:
                raise AnswerExtractionError("Grid dimensions too small to extract answers.")

            _, thresh_img = cv2.threshold(
                cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            question_num = 1
            for col in range(self.cols):
                for row in range(self.rows):
                    if question_num > self.total_questions:
                        break

                    y = row * cell_h
                    x = col * col_w

                    cell_gray = cropped_img[y:y + cell_h, x:x + col_w]
                    cell_thresh = thresh_img[y:y + cell_h, x:x + col_w]

                    filled_counts = []
                    intensities = []

                    for opt in range(self.options):
                        opt_x = opt * opt_w
                        opt_gray = cell_gray[:, opt_x:opt_x + opt_w]
                        opt_thresh = cell_thresh[:, opt_x:opt_x + opt_w]

                        if opt_gray.size == 0 or opt_thresh.size == 0:
                            filled_counts.append(0)
                            intensities.append(255)
                            continue

                        filled = cv2.countNonZero(opt_thresh)
                        filled_counts.append(filled)
                        intensities.append(np.mean(opt_gray))

                    max_filled = max(filled_counts)
                    min_intensity = min(intensities)
                    max_idx = filled_counts.index(max_filled)
                    min_idx = intensities.index(min_intensity)

                    fill_ratio = max_filled / (cell_h * opt_w)
                    is_filled = fill_ratio > 0.1 or min_intensity < 150

                    if is_filled:
                        answer = max_idx + 1 if max_idx == min_idx else None
                    else:
                        answer = None

                    answers.append(answer)
                    question_num += 1

            return answers

        except Exception as e:
            self.logger.exception("Error extracting answers from the sheet.")
            raise AnswerExtractionError(str(e))

    def validate_answers(self, answers):
        try:
            answered = len([a for a in answers if a is not None])
            unanswered = self.total_questions - answered
            formatted = {i + 1: ans for i, ans in enumerate(answers) if ans is not None}

            self.logger.info(f"Answered: {answered}, Unanswered: {unanswered}")
            return {
                "answers": formatted,
                "total": self.total_questions,
                "answered": answered,
                "unanswered": unanswered,
                "raw": answers
            }

        except Exception as e:
            self.logger.exception("Error validating answers.")
            raise MCQDetectorError("Answer validation failed.")
