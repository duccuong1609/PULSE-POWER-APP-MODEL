from pydantic import BaseModel

class RecommendProductRequest(BaseModel):
    product_id: str = "SP000001"
    top_k: int = 10