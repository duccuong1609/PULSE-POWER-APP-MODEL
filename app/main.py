from fastapi import FastAPI
from app.utils.model_loader import load_model_and_mappings
from app.utils.neuMF_model_loader import load_model_neuMF
from app.controllers.customer import recommendation_controller
from app.services.customer.top_k_related_with_customer import RecommendationCustomerService
from app.services.customer.top_k_related_with_customer_neuMF import RecommendationCustomerServiceNeuMF
from app.services.product.top_k_related_with_product_neuMF import RecommendationProductNeuMFService

# ==== 1. Load model và mappings khi startup ====
model, user2idx, item2idx, idx2user, idx2item, device = load_model_and_mappings()
neuMF_model, ckpt, user_enc, item_enc, train_user_pos = load_model_neuMF()

# ==== 2. Khởi tạo service ====
recommend_service = RecommendationCustomerService(model, user2idx, item2idx, idx2user, idx2item, device)
neuMF_recommend_service = RecommendationCustomerServiceNeuMF(neuMF_model, ckpt, user_enc, item_enc, train_user_pos)
product_recommend_service = RecommendationProductNeuMFService(neuMF_model, item_enc)

# ==== 3. Gán service vào controller ====
recommendation_controller.recommend_service = recommend_service
recommendation_controller.neuMF_recommend_service = neuMF_recommend_service
recommendation_controller.product_recommend_service = product_recommend_service

# ==== 4. Khởi tạo FastAPI và include router ====
app = FastAPI(title="PULSE MODEL API", version="1.0.0")
app.include_router(recommendation_controller.router, prefix="/api")