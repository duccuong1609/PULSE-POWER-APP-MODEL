from fastapi import FastAPI
from app.utils.model_loader import load_model_and_mappings
from app.controllers.customer import recommendation_controller
from app.services.customer.top_k_related_with_customer import RecommendationCustomerService

# ==== 1. Load model và mappings khi startup ====
model, user2idx, item2idx, idx2user, idx2item, device = load_model_and_mappings()

# ==== 2. Khởi tạo service ====
recommend_service = RecommendationCustomerService(model, user2idx, item2idx, idx2user, idx2item, device)

# ==== 3. Gán service vào controller ====
recommendation_controller.recommend_service = recommend_service

# ==== 4. Khởi tạo FastAPI và include router ====
app = FastAPI(title="PULSE MODEL API", version="1.0.0")
app.include_router(recommendation_controller.router, prefix="/api")