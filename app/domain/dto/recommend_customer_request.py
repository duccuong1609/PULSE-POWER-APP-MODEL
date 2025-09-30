from pydantic import BaseModel

class RecommendRequest(BaseModel):
    user_id: str = "KH000002"
    top_k: int = 10