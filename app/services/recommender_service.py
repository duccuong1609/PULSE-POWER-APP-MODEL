import json
from app.utils.model_loader import ModelLoader
from app.domain.dto.pydantic_models import RecommendationResponse

class RecommenderService:
    
    @staticmethod
    def get_recommendations(cart_items: list, top_k: int, model_name: str) -> RecommendationResponse:
        model = ModelLoader.get_model(model_name)
        
        if not model:
            model = ModelLoader.get_model("hybrid")
            model_name = "hybrid (fallback)"
            if not model:
                return {
                    "status": "error",
                    "message": "System not ready or models missing",
                    "recommendations": []
                }

        try:
            json_result_str = model.recommend(cart_items, top_k=top_k)
            result_dict = json.loads(json_result_str)
            
            result_dict['model_used'] = model_name
            return result_dict
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                "status": "error", 
                "message": str(e), 
                "recommendations": [],
                "input_cart": cart_items,
                "model_used": model_name
            }