from fastapi import APIRouter, HTTPException
from app.domain.dto.recommend_customer_request import RecommendRequest
from app.domain.dto.recommend_customer_info import RecommendInfo
from app.domain.dto.error_info import ErrorInfo
from fastapi.responses import JSONResponse
from app.services.customer.top_k_related_with_customer import RecommendationCustomerService
from app.services.customer.top_k_related_with_customer_neuMF import RecommendationCustomerServiceNeuMF
from app.services.product.top_k_related_with_product_neuMF import RecommendationProductNeuMFService
from app.domain.dto.recommend_product_info import RecommendProductInfo
from app.domain.dto.recommend_product_request import RecommendProductRequest

router = APIRouter()

recommend_service: RecommendationCustomerService = None 
neuMF_recommend_service: RecommendationCustomerServiceNeuMF = None
product_recommend_service: RecommendationProductNeuMFService = None

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
        
@router.post("/recommend_neuMF", response_model=RecommendInfo)
def recommend_neuMF(req: RecommendRequest):
    if neuMF_recommend_service is None:
        raise HTTPException(
            status_code=500,
            detail="NeuMF Recommendation service not initialized"
        )
    
    try:
        recommend_info = neuMF_recommend_service.recommend_user_topk_by_customer_id(
            ma_kh=req.user_id, topK=req.top_k
        )

        if not recommend_info.recommendations:
            return JSONResponse(
                status_code=404,
                content=ErrorInfo(
                    status=404,
                    message="No recommendations found",
                    details=f"No recommendations for user {req.user_id}"
                ).model_dump()
            )

        return recommend_info 
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorInfo(
                status=500,
                message="Internal server error",
                details=str(e)
            ).model_dump()
        )
        
        
@router.post("/recommend_neuMF_product", response_model=RecommendProductInfo)
def recommend_neuMF_product(req: RecommendProductRequest):
    if product_recommend_service is None:
        raise HTTPException(
            status_code=500,
            detail="Product NeuMF Recommendation service not initialized"
        )
    
    try:
        recommend_info = product_recommend_service.recommend_similar_items_by_mahang(
            ma_hang=req.product_id, topK=req.top_k
        )

        if not recommend_info.recommendations:
            return JSONResponse(
                status_code=404,
                content=ErrorInfo(
                    status=404,
                    message="No recommendations found",
                    details=f"No recommendations for item {req.user_id}"
                ).model_dump()
            )

        return recommend_info 
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorInfo(
                status=500,
                message="Internal server error",
                details=str(e)
            ).model_dump()
        )