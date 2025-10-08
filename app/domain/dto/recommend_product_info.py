from pydantic import BaseModel
from typing import List

class OrderInfo(BaseModel):
    product_id: str
    score: float
    rank: int

class RecommendProductInfo(BaseModel):
    product_id: str
    recommendations: List[OrderInfo]