from typing import List, Dict, Any, Optional
from pydantic import BaseModel
class State(BaseModel):
    user_message: Optional[str] = None
    chat_history: List[Dict[str, Any]] = []
    data: Dict[str, Any]
    analysis_data: Optional[Dict[str, Any]] = None
    analyst_signals: Optional[Dict[str, Any]] = None
    secrets: Optional[Dict[str, Any]] = None
