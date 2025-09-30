from pydantic import BaseModel
from typing import List

class OrderInfo(BaseModel):
    MaHang: str
    Score: float
    Rank: int

class RecommendInfo(BaseModel):
    user_id: str
    recommendations: List[OrderInfo]