class MCQDetectorError(Exception):
    """Base class for all MCQ Detector errors"""
    pass

class ImageNotFoundError(MCQDetectorError):
    pass

class ImageProcessingError(MCQDetectorError):
    pass

class GridDetectionError(MCQDetectorError):
    pass

class AnswerExtractionError(MCQDetectorError):
    pass
