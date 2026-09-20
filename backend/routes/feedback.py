from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger("weathergpt.feedback")

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackRequest(BaseModel):
    message_id: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    language: Optional[str] = "en"

@router.post("")
def submit_feedback(request: FeedbackRequest):
    logger.info(f"Received feedback: rating={request.rating}, lang={request.language}, comment={request.comment}")
    return {
        "success": True,
        "message": "Feedback recorded successfully. Thank you!",
        "timestamp": datetime.now().isoformat()
    }
