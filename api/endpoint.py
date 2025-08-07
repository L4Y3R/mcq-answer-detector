from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from configs.logger import setup_logger
from detector.processor import MCQAnswerDetector
from exceptions.exceptions import MCQDetectorError

router = APIRouter()
logger = setup_logger()
processor = MCQAnswerDetector(logger)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image to disk
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {file.filename}")
        original, gray, _ = processor.preprocess_image(file_path)
        cropped = processor.get_grid_region(gray)
        answers = processor.extract_answers(cropped)
        result = processor.validate_answers(answers)

        # Build pretty printed result string
        output_lines = ["ANSWER SHEET RESULTS", "=" * 50]
        for q in range(1, result["total"] + 1):
            ans = result["answers"].get(q, "-")
            output_lines.append(f"Q{q:2d}: {ans}")
            if q % 10 == 0:
                output_lines.append("")
        pretty_output = "\n".join(output_lines)

        return JSONResponse({
            "status": "success",
            "answers": result["answers"],
            "answered": result["answered"],
            "unanswered": result["unanswered"],
            "total_questions": result["total"]
        })

    except MCQDetectorError as e:
        logger.error(f"[MCQ Error] {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    
@router.delete("/clear-uploads")
async def clear_uploads():
    try:
        removed_files = []

        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_files.append(filename)

        logger.info(f"Cleared uploads folder. Removed: {removed_files}")

        return JSONResponse({
            "status": "success",
            "message": f"{len(removed_files)} file(s) removed.",
            "files_removed": removed_files
        })

    except Exception as e:
        logger.exception("Error while clearing uploads folder.")
        return JSONResponse(status_code=500, content={"error": "Failed to clear uploads folder."})

