from pydantic import BaseModel

class ErrorInfo(BaseModel):
    status: int
    message: str
    details: str = None