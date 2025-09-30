from fastapi import APIRouter, HTTPException
from app.domain.dto.recommend_customer_request import RecommendRequest
from app.domain.dto.recommend_customer_info import RecommendInfo
from app.domain.dto.error_info import ErrorInfo
from fastapi.responses import JSONResponse
from app.services.customer.top_k_related_with_customer import RecommendationCustomerService

router = APIRouter()

recommend_service: RecommendationCustomerService = None 

@router.post("/recommend", response_model=RecommendInfo)
def recommend(req: RecommendRequest):
    if recommend_service is None:
        raise HTTPException(
            status_code=500,
            detail="Recommendation service not initialized"
        )
    
    try:
        items = recommend_service.recommend_top_k(req.user_id, top_k=req.top_k)
        if not items:
            return JSONResponse(
                status_code=404,
                content=ErrorInfo(status=404, message="No recommendations found", details=f"No recommendations for user {req.user_id}").model_dump() 
            )
        return RecommendInfo(user_id=req.user_id, recommendations=items)
    
    except Exception as e:
        return JSONResponse(
                status_code=500,
                content=ErrorInfo(status=500, message="Internal server error", details=str(e)).model_dump()
            )