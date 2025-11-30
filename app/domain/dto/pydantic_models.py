from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class CartRequest(BaseModel):
    cart_items: List[str] = Field(
        default=["SP000007", "SP000013"],
        description="Danh sách mã sản phẩm đang có trong giỏ hàng",
        examples=[["SP000007", "SP000013"]]
    )
    top_k: int = Field(
        default=5, 
        description="Số lượng sản phẩm muốn gợi ý"
    )

class ProductRecommendation(BaseModel):
    product_id: str
    score: float

class RecommendationResponse(BaseModel):
    status: str
    input_cart: List[str]
    recommendations: List[ProductRecommendation]
    model_used: str
    
class ModelName(str, Enum):
    HYBRID = "hybrid"
    EASE = "ease"
    ALS = "als"
    KNN = "knn"
    SLIM = "slim"
    SVD = "svd"