import os
import cv2
import numpy as np
import logging
from configs.config import Config
from exceptions.exceptions import (
    ImageNotFoundError,
    ImageProcessingError,
    GridDetectionError,
    AnswerExtractionError,
    MCQDetectorError
)


class MCQAnswerDetectorV2:
    def __init__(self, logger):
        self.logger = logger
        self.total_questions = Config.TOTAL_QUESTIONS
        self.rows = Config.ROWS
        self.cols = Config.COLS
        self.options = Config.OPTIONS

        # Improved parameters
        self.min_contour_area = 50
        self.max_contour_area = 2000
        self.circularity_threshold = 0.3
        self.fill_threshold = 0.15  # Percentage of circle that needs to be filled

    def preprocess_image(self, image_path):
        try:
            if not image_path or not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                raise ImageNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                self.logger.error("Failed to read image.")
                raise ImageProcessingError("Could not decode image.")

            # Improve image quality
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)

            # Light blur to smooth noise
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)

            debug_path = f"{Config.LOG_DIR}/debug_preprocessed.png"
            cv2.imwrite(debug_path, blurred)
            self.logger.info(f"Preprocessed image saved at {debug_path}.")

            return image, gray, blurred

        except Exception as e:
            self.logger.exception("Error during image preprocessing.")
            raise ImageProcessingError(str(e))

    def detect_grid_automatically(self, gray_image):
        """Automatically detect the grid region using contour detection"""
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest rectangular contour (likely the answer grid)
            largest_area = 0
            best_rect = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # If it's roughly rectangular and large enough
                    if len(approx) >= 4 and area > gray_image.shape[0] * gray_image.shape[1] * 0.1:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h

                        # Check if it looks like our answer grid (wider than tall)
                        if 0.8 < aspect_ratio < 2.5:
                            largest_area = area
                            best_rect = (x, y, w, h)

            if best_rect:
                x, y, w, h = best_rect
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray_image.shape[1] - x, w + 2 * padding)
                h = min(gray_image.shape[0] - y, h + 2 * padding)

                self.logger.info(f"Auto-detected grid: x={x}, y={y}, w={w}, h={h}")
                return gray_image[y:y + h, x:x + w]
            else:
                # Fall back to manual cropping if auto-detection fails
                return self.get_grid_region_manual(gray_image)

        except Exception as e:
            self.logger.warning(f"Auto grid detection failed: {e}, falling back to manual")
            return self.get_grid_region_manual(gray_image)

    def get_grid_region_manual(self, gray_image):
        """Manual grid cropping as fallback"""
        try:
            h, w = gray_image.shape
            top, bottom = int(h * 0.25), int(h * 0.95)
            left, right = int(w * 0.06), int(w * 0.985)
            cropped = gray_image[top:bottom, left:right]

            self.logger.info(f"Manual cropped grid: top={top}, bottom={bottom}, left={left}, right={right}")
            return cropped

        except Exception as e:
            self.logger.exception("Error cropping the grid region.")
            raise GridDetectionError("Grid region could not be determined.")

    def detect_circles_in_cell(self, cell_gray, cell_thresh):
        """Detect circles (answer options) within a cell using HoughCircles"""
        circles = cv2.HoughCircles(
            cell_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(cell_gray.shape[1] / 6),  # Minimum distance between circles
            param1=50,
            param2=20,
            minRadius=int(min(cell_gray.shape) / 15),
            maxRadius=int(min(cell_gray.shape) / 8)
        )

        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort circles by x coordinate (left to right)
            circles = sorted(circles, key=lambda c: c[0])
            detected_circles = circles[:self.options]  # Take only the expected number

        return detected_circles

    def analyze_circle_filling(self, cell_gray, circle):
        """Analyze if a circle is filled/marked"""
        x, y, r = circle

        # Create a mask for the circle
        mask = np.zeros(cell_gray.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Extract the circular region
        circle_region = cv2.bitwise_and(cell_gray, cell_gray, mask=mask)
        circle_pixels = circle_region[mask > 0]

        if len(circle_pixels) == 0:
            return False, 0

        # Calculate statistics
        mean_intensity = np.mean(circle_pixels)
        std_intensity = np.std(circle_pixels)
        min_intensity = np.min(circle_pixels)

        # Also check for edge pixels (crosses or heavy marks)
        edges = cv2.Canny(circle_region, 50, 150)
        edge_pixels = np.sum(edges[mask > 0] > 0)
        edge_ratio = edge_pixels / len(circle_pixels)

        # Multiple criteria for marking detection
        is_filled = (
                mean_intensity < 180 or  # Dark filling
                min_intensity < 100 or  # Very dark spots
                edge_ratio > 0.1  # Significant edges (crosses)
        )

        confidence = 0
        if mean_intensity < 150:
            confidence += 0.4
        if min_intensity < 80:
            confidence += 0.3
        if edge_ratio > 0.15:
            confidence += 0.3
        if std_intensity > 30:  # High variance suggests marking
            confidence += 0.2

        return is_filled, confidence

    def extract_answers(self, cropped_img):
        try:
            answers = []
            h, w = cropped_img.shape
            cell_h = h // self.rows
            col_w = w // self.cols

            if cell_h == 0 or col_w == 0:
                raise AnswerExtractionError("Grid dimensions too small to extract answers.")

            # Use adaptive thresholding for better results
            thresh_img = cv2.adaptiveThreshold(
                cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Save debug image
            debug_path = f"{Config.LOG_DIR}/debug_thresh.png"
            cv2.imwrite(debug_path, thresh_img)

            question_num = 1
            for col in range(self.cols):
                for row in range(self.rows):
                    if question_num > self.total_questions:
                        break

                    y = row * cell_h
                    x = col * col_w

                    cell_gray = cropped_img[y:y + cell_h, x:x + col_w]
                    cell_thresh = thresh_img[y:y + cell_h, x:x + col_w]

                    if cell_gray.size == 0:
                        answers.append(None)
                        question_num += 1
                        continue

                    # Try to detect circles automatically
                    circles = self.detect_circles_in_cell(cell_gray, cell_thresh)

                    if len(circles) >= self.options:
                        # Use detected circles
                        option_results = []
                        for i, circle in enumerate(circles[:self.options]):
                            is_filled, confidence = self.analyze_circle_filling(cell_gray, circle)
                            option_results.append((i + 1, is_filled, confidence))
                    else:
                        # Fall back to grid-based analysis
                        option_results = []
                        opt_w = col_w // self.options

                        for opt in range(self.options):
                            opt_x = opt * opt_w
                            opt_gray = cell_gray[:, opt_x:opt_x + opt_w]

                            if opt_gray.size == 0:
                                option_results.append((opt + 1, False, 0))
                                continue

                            # Simple analysis for fallback
                            mean_intensity = np.mean(opt_gray)
                            min_intensity = np.min(opt_gray)

                            is_filled = mean_intensity < 180 or min_intensity < 100
                            confidence = (200 - mean_intensity) / 200 if is_filled else 0

                            option_results.append((opt + 1, is_filled, confidence))

                    # Determine the answer based on confidence scores
                    filled_options = [(opt, conf) for opt, filled, conf in option_results if filled]

                    if len(filled_options) == 1:
                        # Only one option marked - ideal case
                        answer = filled_options[0][0]
                        self.logger.debug(f"Question {question_num}: Single option detected -> Answer: {answer}")
                    elif len(filled_options) > 1:
                        # Multiple options marked - choose the one with highest confidence
                        answer = max(filled_options, key=lambda x: x[1])[0]
                        filled_answers = [opt for opt, _ in filled_options]
                        self.logger.warning(
                            f"Question {question_num}: Multiple options detected {filled_answers}, chose highest confidence -> Answer: {answer}")
                    else:
                        # No option marked
                        answer = None
                        self.logger.debug(f"Question {question_num}: No option detected -> Answer: None")

                    answers.append(answer)

                    # Enhanced debug logging for all questions
                    if self.logger.isEnabledFor(logging.DEBUG):
                        confidence_info = [(opt, f"{conf:.2f}" if conf > 0 else "0.00") for opt, filled, conf in
                                           option_results if filled]
                        self.logger.debug(
                            f"Q{question_num}: Filled options with confidence: {confidence_info} -> Final Answer: {answer}")

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

            # Additional validation and detailed answer logging
            option_counts = {i: 0 for i in range(1, 6)}
            answer_details = []

            for i, ans in enumerate(answers):
                if ans is not None:
                    option_counts[ans] = option_counts.get(ans, 0) + 1
                    answer_details.append(f"Q{i + 1}:{ans}")

            # Log all detected answers in groups of 10 for readability
            for i in range(0, len(answer_details), 10):
                group = answer_details[i:i + 10]
                self.logger.info(f"Answers {i + 1}-{min(i + 10, len(answer_details))}: {' '.join(group)}")

            self.logger.info(f"Option distribution: {option_counts}")

            return {
                "answers": formatted,
                "total": self.total_questions,
                "answered": answered,
                "unanswered": unanswered,
                "raw": answers,
                "option_distribution": option_counts
            }

        except Exception as e:
            self.logger.exception("Error validating answers.")
            raise MCQDetectorError("Answer validation failed.")

    def process_image(self, image_path):
        """Main method to process an MCQ image and extract answers"""
        try:
            # Preprocess image
            original, gray, processed = self.preprocess_image(image_path)

            # Detect grid region
            grid_region = self.detect_grid_automatically(processed)

            # Save cropped grid for debugging
            debug_path = f"{Config.LOG_DIR}/debug_grid.png"
            cv2.imwrite(debug_path, grid_region)

            # Extract answers
            answers = self.extract_answers(grid_region)

            # Validate and format results
            results = self.validate_answers(answers)

            return results

        except Exception as e:
            self.logger.exception("Error processing MCQ image.")
            raise MCQDetectorError(f"Failed to process image: {str(e)}")