import cv2
import numpy as np
import logging
import os
from imutils import contours


class Config:
    TOTAL_QUESTIONS = 50
    OPTIONS = 5
    # This is now less critical, but helps in sorting questions correctly
    # if they are in multiple columns.
    QUESTIONS_PER_COLUMN = 25
    LOG_DIR = "logs"
    # A bubble must be at least 35% filled to be considered marked.
    # Adjust this if marks are very light or very heavy.
    BUBBLE_FILLED_THRESHOLD = 0.35


class MCQAnswerDetectorV3:
    """
    Final robust version of MCQ Answer Detector.
    This version finds all bubble contours first and then uses their positions
    to infer the grid structure, making it highly resilient to layout variations.
    """

    def __init__(self, logger):
        self.logger = logger
        # Initialize all attributes from the Config class
        self.total_questions = Config.TOTAL_QUESTIONS
        self.options = Config.OPTIONS
        self.questions_per_col = Config.QUESTIONS_PER_COLUMN
        self.log_dir = Config.LOG_DIR
        self.fill_threshold = Config.BUBBLE_FILLED_THRESHOLD

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _log_and_save_debug_image(self, image, filename_suffix):
        """Helper function to save debug images."""
        debug_path = os.path.join(self.log_dir, f"debug_{filename_suffix}.png")
        cv2.imwrite(debug_path, image)
        self.logger.debug(f"Saved debug image: {debug_path}")

    def process_image(self, image_path: str) -> dict:
        """Main processing pipeline to detect all answers from an image file."""
        try:
            self.logger.info(f"Starting V6 processing for image: {image_path}")

            # 1. Load the image and get a clean, top-down, binary view of the grid
            warped_binary = self._find_and_warp_grid(image_path)

            # 2. Find all bubble contours and organize them into question rows
            question_rows = self._find_and_organize_bubbles(warped_binary)

            if len(question_rows) < self.total_questions:
                self.logger.warning(
                    f"Found {len(question_rows)} question rows, but expected {self.total_questions}. "
                    "Results may be incomplete. Check for poor scans or incorrect bubble detection."
                )

            # 3. Analyze each row to find the marked answer
            answers = self._analyze_rows_for_answers(question_rows, warped_binary)

            # 4. Format and return final results
            return self._format_results(answers)

        except Exception as e:
            self.logger.exception(f"A critical error occurred in the main V6 pipeline: {e}")
            return {"error": str(e)}

    def _find_and_warp_grid(self, image_path: str) -> np.ndarray:
        """Loads, preprocesses, and returns a binary top-down view of the answer grid."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image could not be read from {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced blur kernel for sharper edges

        # This thresholding is for finding the main grid border
        edged = cv2.Canny(blurred, 75, 200)
        self._log_and_save_debug_image(edged, "1_edged")

        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        grid_contour = None
        doc_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Find the first contour with 4 corners, which is likely the grid
        for c in doc_cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                grid_contour = approx
                break

        if grid_contour is None:
            raise ValueError("Could not find the 4-corner answer grid contour.")

        debug_image_contour = image.copy()
        cv2.drawContours(debug_image_contour, [grid_contour], -1, (0, 255, 0), 3)
        self._log_and_save_debug_image(debug_image_contour, "2_grid_contour")

        # Warp both the grayscale and the original color image
        warped_gray = self._four_point_transform(gray, grid_contour.reshape(4, 2))

        # Now, apply a robust threshold to the warped image to get a clean binary version
        # This is the image we will use for all subsequent analysis
        binary_warped = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        self._log_and_save_debug_image(binary_warped, "3_binary_warped")
        return binary_warped

    def _find_and_organize_bubbles(self, binary_warped: np.ndarray) -> list:
        """Find all bubble contours and group them by question row."""
        cnts, _ = cv2.findContours(binary_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_contours = []
        # Filter contours to find only the circular answer bubbles
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            # Flexible criteria for a bubble
            if 15 < w < 40 and 15 < h < 40 and 0.8 < aspect_ratio < 1.2:
                bubble_contours.append(c)

        self.logger.info(f"Found {len(bubble_contours)} potential bubble contours.")

        # Sort the bubbles into rows. First sort by columns (for multi-column sheets),
        # then by their y-coordinate.
        question_rows = []
        if bubble_contours:
            # Sort all bubbles from top to bottom
            sorted_bubbles = contours.sort_contours(bubble_contours, method="top-to-bottom")[0]

            # Group the bubbles into rows of 5
            for i in range(0, len(sorted_bubbles), self.options):
                # Grab a chunk of contours that should correspond to a single question
                row = sorted_bubbles[i: i + self.options]

                # A valid row must have exactly 5 bubbles
                if len(row) == self.options:
                    # Sort the bubbles in the row by their x-coordinate (left to right)
                    # This ensures option 1 is always first
                    left_to_right_row = contours.sort_contours(row, method="left-to-right")[0]
                    question_rows.append(left_to_right_row)

        return question_rows

    def _analyze_rows_for_answers(self, question_rows: list, binary_warped: np.ndarray) -> dict:
        """Iterate through sorted rows, find the marked bubble, and assign the answer."""
        answers = {}
        # As question_rows is now sorted top-to-bottom and left-to-right,
        # we can just assign question numbers sequentially.
        for (q_num, row_contours) in enumerate(question_rows, 1):
            if q_num > self.total_questions:
                break

            marked_option = None
            max_fill_ratio = 0.0

            # Find the bubble in the row with the most filled pixels
            for i, c in enumerate(row_contours):
                # Create a mask for the current bubble
                mask = np.zeros(binary_warped.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)

                # Get the number of filled pixels within the bubble
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(binary_warped, binary_warped, mask=mask))

                # Calculate fill ratio against the bubble's own area
                total_area = cv2.contourArea(c)
                fill_ratio = filled_pixels / total_area if total_area > 0 else 0

                if fill_ratio > max_fill_ratio:
                    max_fill_ratio = fill_ratio
                    marked_option = i + 1  # +1 because options are 1-indexed

            # Only record an answer if a bubble was filled above the threshold
            if marked_option is not None and max_fill_ratio > self.fill_threshold:
                answers[q_num] = marked_option

        return answers

    def _format_results(self, answers: dict) -> dict:
        """Formats the final dictionary of results."""
        answered_count = len(answers)
        rate = answered_count / self.total_questions if self.total_questions > 0 else 0
        self.logger.info(f"Final Detection Rate: {rate:.1%}")
        return {
            "answers": answers,
            "total_questions": self.total_questions,
            "answered": answered_count,
            "unanswered": self.total_questions - answered_count,
            "detection_rate": rate
        }

    # Helper functions for perspective transform (unchanged from previous versions)
    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
