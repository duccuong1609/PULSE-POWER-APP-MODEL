from fastapi import APIRouter, Query
from app.domain.dto.pydantic_models import CartRequest, RecommendationResponse, ModelName
from app.services.recommender_service import RecommenderService

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_products(
    request: CartRequest,
    model_name: ModelName = Query(..., description="Chọn mô hình gợi ý")
):
    """
    API gợi ý sản phẩm.
    - **Body**: { "cart_items": [...], "top_k": 5 }
    - **Query Param**: ?model_name=hybrid (Chọn model tại đây)
    """
    
    if not request.cart_items:
        return {
            "status": "warning",
            "input_cart": [],
            "recommendations": [],
            "model_used": "none",
            "message": "Giỏ hàng rỗng"
        }

    result = RecommenderService.get_recommendations(
        cart_items=request.cart_items,
        top_k=request.top_k,
        model_name=model_name.value
    )
    
    return result