from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel

class InterruptInfo(BaseModel):
    reason: Literal["AMBIGUOUS_CHOICE", "MISSING_INFO", "CONFIRMATION_REQUIRED"]
    question: str

class GUI_OUTPUT(BaseModel):
    thought: str
    action: Literal["CLICK", "LONG_CLICK", "SWIPE", "INPUT", "BACK", "HOME", "INTERRUPT"]
    box_2d: List[float]
    interrupt: Optional[InterruptInfo]