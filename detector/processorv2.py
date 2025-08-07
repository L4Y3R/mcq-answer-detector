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

        # Optimized parameters for the specific answer sheet format
        self.min_contour_area = 80
        self.max_contour_area = 3000
        self.circularity_threshold = 0.4
        self.fill_threshold = 0.20

        # Grey/black detection parameters
        self.dark_intensity_threshold = 160  # Values below this are considered dark
        self.dark_pixel_ratio_threshold = 0.15  # Minimum ratio of dark pixels to consider filled

    def preprocess_image(self, image_path):
        try:
            if not image_path or not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                raise ImageNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                self.logger.error("Failed to read image.")
                raise ImageProcessingError("Could not decode image.")

            # Resize if image is too large (for consistent processing)
            height, width = image.shape[:2]
            if width > 2000 or height > 2000:
                scale_factor = min(2000 / width, 2000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Bilateral filter to reduce noise while keeping edges sharp
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            debug_path = f"{Config.LOG_DIR}/debug_preprocessed_v3.png"
            cv2.imwrite(debug_path, denoised)
            self.logger.info(f"Preprocessed image saved at {debug_path}.")

            return image, gray, denoised, None  # No longer returning HSV image

        except Exception as e:
            self.logger.exception("Error during image preprocessing.")
            raise ImageProcessingError(str(e))

    def detect_dark_regions(self, gray_image):
        """Detect dark-filled bubbles using intensity thresholding"""
        # Create mask for dark regions
        _, dark_mask = cv2.threshold(gray_image, self.dark_intensity_threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        debug_path = f"{Config.LOG_DIR}/debug_dark_mask.png"
        cv2.imwrite(debug_path, dark_mask)

        return dark_mask

    def detect_grid_region_precise(self, gray_image, original_image):
        """More precise grid detection using contours and layout analysis"""
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 3
            )

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for the main grid border
            h, w = gray_image.shape
            min_area = (h * w) * 0.15  # At least 15% of image
            max_area = (h * w) * 0.8  # At most 80% of image

            best_rect = None
            best_score = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Get bounding rectangle
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    aspect_ratio = w_rect / h_rect

                    # Score based on area, aspect ratio, and position
                    area_score = min(area / (h * w * 0.5), 1.0)  # Normalize to max 1.0
                    aspect_score = 1.0 if 1.2 < aspect_ratio < 2.0 else 0.5
                    position_score = 1.0 if y > h * 0.1 and y < h * 0.4 else 0.7  # Should be in upper-middle

                    total_score = area_score * aspect_score * position_score

                    if total_score > best_score:
                        best_score = total_score
                        best_rect = (x, y, w_rect, h_rect)

            if best_rect:
                x, y, w_rect, h_rect = best_rect
                # Fine-tune the boundaries by looking for grid lines
                grid_region = gray_image[y:y + h_rect, x:x + w_rect]

                self.logger.info(f"Detected grid region: x={x}, y={y}, w={w_rect}, h={h_rect}")
                debug_path = f"{Config.LOG_DIR}/debug_detected_grid.png"
                cv2.imwrite(debug_path, grid_region)

                return grid_region, (x, y, w_rect, h_rect)
            else:
                # Fallback to manual estimation based on typical answer sheet layout
                return self.get_grid_region_manual_v3(gray_image)

        except Exception as e:
            self.logger.warning(f"Grid detection failed: {e}")
            return self.get_grid_region_manual_v3(gray_image)

    def get_grid_region_manual_v3(self, gray_image):
        """Improved manual grid extraction based on answer sheet analysis"""
        try:
            h, w = gray_image.shape

            # Based on the answer sheet image, the grid appears to be roughly:
            # - Top: around 25-30% from top
            # - Bottom: around 85-90% from top
            # - Left: around 5-8% from left
            # - Right: around 92-95% from left

            top = int(h * 0.28)
            bottom = int(h * 0.87)
            left = int(w * 0.06)
            right = int(w * 0.94)

            cropped = gray_image[top:bottom, left:right]
            bounds = (left, top, right - left, bottom - top)

            self.logger.info(f"Manual grid crop: top={top}, bottom={bottom}, left={left}, right={right}")

            return cropped, bounds

        except Exception as e:
            self.logger.exception("Error in manual grid cropping.")
            raise GridDetectionError("Grid region could not be determined.")

    def detect_bubbles_in_cell(self, cell_image, dark_mask_cell):
        """Detect and analyze bubbles within a single cell"""
        bubbles = []

        # Use HoughCircles to detect circular shapes
        circles = cv2.HoughCircles(
            cell_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(cell_image.shape[1] / 6),
            param1=40,
            param2=25,
            minRadius=max(5, int(min(cell_image.shape) / 20)),
            maxRadius=int(min(cell_image.shape) / 6)
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by x-coordinate (left to right)
            circles = sorted(circles, key=lambda c: c[0])

            for i, (x, y, r) in enumerate(circles[:self.options]):
                # Check if this bubble is filled using dark mask
                mask = np.zeros(cell_image.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)

                # Check dark filling
                dark_pixels = cv2.bitwise_and(dark_mask_cell, dark_mask_cell, mask=mask)
                dark_ratio = np.sum(dark_pixels > 0) / np.sum(mask > 0)

                # Check grayscale intensity
                circle_pixels = cell_image[mask > 0]
                mean_intensity = np.mean(circle_pixels) if len(circle_pixels) > 0 else 255

                # Combine dark detection and intensity analysis
                is_filled = dark_ratio > self.dark_pixel_ratio_threshold or mean_intensity < self.dark_intensity_threshold
                confidence = dark_ratio * 2 + (255 - mean_intensity) / 255

                bubbles.append({
                    'option': i + 1,
                    'position': (x, y, r),
                    'is_filled': is_filled,
                    'confidence': confidence,
                    'dark_ratio': dark_ratio,
                    'mean_intensity': mean_intensity
                })

        # If no circles detected, use grid-based approach
        if not bubbles:
            bubbles = self.fallback_grid_analysis(cell_image, dark_mask_cell)

        return bubbles

    def fallback_grid_analysis(self, cell_image, dark_mask_cell):
        """Fallback method when circle detection fails"""
        bubbles = []
        cell_width = cell_image.shape[1]
        option_width = cell_width // self.options

        for i in range(self.options):
            x_start = i * option_width
            x_end = min(x_start + option_width, cell_width)

            option_region = cell_image[:, x_start:x_end]
            dark_region = dark_mask_cell[:, x_start:x_end]

            if option_region.size > 0:
                mean_intensity = np.mean(option_region)
                dark_ratio = np.sum(dark_region > 0) / dark_region.size

                is_filled = dark_ratio > self.dark_pixel_ratio_threshold or mean_intensity < self.dark_intensity_threshold
                confidence = dark_ratio * 1.5 + (255 - mean_intensity) / 255

                bubbles.append({
                    'option': i + 1,
                    'position': (x_start + option_width // 2, cell_image.shape[0] // 2, option_width // 4),
                    'is_filled': is_filled,
                    'confidence': confidence,
                    'dark_ratio': dark_ratio,
                    'mean_intensity': mean_intensity
                })

        return bubbles

    def extract_answers_v3(self, grid_image, dark_mask, grid_bounds):
        """Enhanced answer extraction with better cell detection"""
        try:
            answers = []
            h, w = grid_image.shape

            # Calculate cell dimensions
            cell_height = h // self.rows
            cell_width = w // self.cols

            if cell_height < 20 or cell_width < 50:
                raise AnswerExtractionError("Grid cells too small for reliable detection.")

            self.logger.info(f"Grid dimensions: {w}x{h}, Cell size: {cell_width}x{cell_height}")

            # Extract dark mask for grid region
            x_offset, y_offset = grid_bounds[0], grid_bounds[1]
            grid_dark_mask = dark_mask[y_offset:y_offset + h, x_offset:x_offset + w]

            question_num = 1

            for col in range(self.cols):
                for row in range(self.rows):
                    if question_num > self.total_questions:
                        break

                    # Calculate cell boundaries with small margins
                    y1 = row * cell_height + 2
                    y2 = (row + 1) * cell_height - 2
                    x1 = col * cell_width + 2
                    x2 = (col + 1) * cell_width - 2

                    # Extract cell
                    cell_image = grid_image[y1:y2, x1:x2]
                    cell_dark_mask = grid_dark_mask[y1:y2, x1:x2] if grid_dark_mask.shape == grid_image.shape else np.zeros_like(cell_image)

                    if cell_image.size == 0:
                        answers.append(None)
                        question_num += 1
                        continue

                    # Detect bubbles in this cell
                    bubbles = self.detect_bubbles_in_cell(cell_image, cell_dark_mask)

                    # Determine the answer
                    filled_bubbles = [b for b in bubbles if b['is_filled']]

                    if len(filled_bubbles) == 1:
                        answer = filled_bubbles[0]['option']
                        confidence = filled_bubbles[0]['confidence']
                        self.logger.debug(f"Q{question_num}: Single answer {answer} (conf: {confidence:.2f})")

                    elif len(filled_bubbles) > 1:
                        # Multiple answers - choose highest confidence
                        best_bubble = max(filled_bubbles, key=lambda x: x['confidence'])
                        answer = best_bubble['option']
                        self.logger.warning(f"Q{question_num}: Multiple answers detected, chose {answer}")

                    else:
                        # No clear answer
                        # Check if any bubble has moderate confidence
                        moderate_bubbles = [b for b in bubbles if b['confidence'] > 0.3]
                        if moderate_bubbles:
                            best_bubble = max(moderate_bubbles, key=lambda x: x['confidence'])
                            answer = best_bubble['option']
                            self.logger.debug(f"Q{question_num}: Moderate confidence answer {answer}")
                        else:
                            answer = None
                            self.logger.debug(f"Q{question_num}: No answer detected")

                    answers.append(answer)

                    # Debug logging
                    if self.logger.isEnabledFor(logging.DEBUG):
                        bubble_info = [(b['option'], f"D:{b['dark_ratio']:.2f}", f"I:{b['mean_intensity']:.0f}",
                                      f"C:{b['confidence']:.2f}") for b in bubbles if b['is_filled']]
                        self.logger.debug(f"Q{question_num}: Filled bubbles: {bubble_info} -> Answer: {answer}")

                    question_num += 1

            return answers

        except Exception as e:
            self.logger.exception("Error extracting answers from the sheet.")
            raise AnswerExtractionError(str(e))

    def validate_answers_v3(self, answers):
        """Enhanced answer validation with more detailed reporting"""
        try:
            answered = len([a for a in answers if a is not None])
            unanswered = self.total_questions - answered
            formatted = {i + 1: ans for i, ans in enumerate(answers) if ans is not None}

            # Option distribution
            option_counts = {i: 0 for i in range(1, self.options + 1)}
            for ans in answers:
                if ans is not None and 1 <= ans <= self.options:
                    option_counts[ans] += 1

            # Detailed logging
            self.logger.info(
                f"Detection Results: {answered}/{self.total_questions} answered ({answered / self.total_questions * 100:.1f}%)")

            # Log answers in groups for better readability
            for i in range(0, len(answers), 10):
                end_idx = min(i + 10, len(answers))
                answer_group = []
                for j in range(i, end_idx):
                    ans = answers[j] if answers[j] is not None else 'X'
                    answer_group.append(f"Q{j + 1}:{ans}")
                self.logger.info(f"Answers {i + 1}-{end_idx}: {' '.join(answer_group)}")

            self.logger.info(f"Option distribution: {option_counts}")

            # Quality checks
            if answered < self.total_questions * 0.5:
                self.logger.warning("Low detection rate - less than 50% of questions detected")

            # Check for unusual patterns
            dominant_option = max(option_counts, key=option_counts.get)
            if option_counts[dominant_option] > self.total_questions * 0.6:
                self.logger.warning(
                    f"Unusual pattern: Option {dominant_option} dominates with {option_counts[dominant_option]} selections")

            return {
                "answers": formatted,
                "total": self.total_questions,
                "answered": answered,
                "unanswered": unanswered,
                "raw": answers,
                "option_distribution": option_counts,
                "detection_rate": answered / self.total_questions
            }

        except Exception as e:
            self.logger.exception("Error validating answers.")
            raise MCQDetectorError("Answer validation failed.")

    def process_image(self, image_path):
        """Main processing method with enhanced pipeline"""
        try:
            # Preprocess image
            original, gray, processed, _ = self.preprocess_image(image_path)  # No longer using HSV

            # Detect dark regions (filled bubbles)
            dark_mask = self.detect_dark_regions(processed)

            # Detect grid region more precisely
            grid_region, grid_bounds = self.detect_grid_region_precise(processed, original)

            # Save debug images
            debug_path = f"{Config.LOG_DIR}/debug_grid_v3.png"
            cv2.imwrite(debug_path, grid_region)

            # Extract answers using enhanced method
            answers = self.extract_answers_v3(grid_region, dark_mask, grid_bounds)

            # Validate and format results
            results = self.validate_answers_v3(answers)

            return results

        except Exception as e:
            self.logger.exception("Error processing MCQ image.")
            raise MCQDetectorError(f"Failed to process image: {str(e)}")

    def debug_cell_analysis(self, image_path, question_num):
        """Debug method to analyze a specific question cell in detail"""
        try:
            original, gray, processed, _ = self.preprocess_image(image_path)
            dark_mask = self.detect_dark_regions(processed)
            grid_region, grid_bounds = self.detect_grid_region_precise(processed, original)

            h, w = grid_region.shape
            cell_height = h // self.rows
            cell_width = w // self.cols

            # Calculate which cell contains the question
            col = (question_num - 1) // self.rows
            row = (question_num - 1) % self.rows

            y1 = row * cell_height + 2
            y2 = (row + 1) * cell_height - 2
            x1 = col * cell_width + 2
            x2 = (col + 1) * cell_width - 2

            cell_image = grid_region[y1:y2, x1:x2]
            x_offset, y_offset = grid_bounds[0], grid_bounds[1]
            grid_dark_mask = dark_mask[y_offset:y_offset + h, x_offset:x_offset + w]
            cell_dark_mask = grid_dark_mask[y1:y2, x1:x2]

            # Analyze the cell
            bubbles = self.detect_bubbles_in_cell(cell_image, cell_dark_mask)

            # Save debug images
            debug_cell_path = f"{Config.LOG_DIR}/debug_cell_q{question_num}.png"
            debug_dark_path = f"{Config.LOG_DIR}/debug_cell_q{question_num}_dark.png"

            cv2.imwrite(debug_cell_path, cell_image)
            cv2.imwrite(debug_dark_path, cell_dark_mask)

            self.logger.info(f"Question {question_num} debug analysis:")
            for bubble in bubbles:
                self.logger.info(f"  Option {bubble['option']}: Filled={bubble['is_filled']}, "
                               f"Confidence={bubble['confidence']:.2f}, DarkRatio={bubble['dark_ratio']:.2f}, "
                               f"Intensity={bubble['mean_intensity']:.1f}")

            return bubbles

        except Exception as e:
            self.logger.exception(f"Error in debug analysis for question {question_num}")
            return []