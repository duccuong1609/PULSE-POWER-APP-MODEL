import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.domain.dto.recommend_product_info import RecommendProductInfo, OrderInfo

class RecommendationProductNeuMFService:
    def __init__(self, model, item_enc, device=None):
        self.model = model
        self.item_enc = item_enc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embeddings = self.build_item_embeddings()

    def build_item_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            gmf_w = self.model.item_embed_gmf.weight.detach().cpu().numpy()
            mlp_w = self.model.item_embed_mlp.weight.detach().cpu().numpy()
            item_repr = np.concatenate([gmf_w, mlp_w], axis=1)
            norms = np.linalg.norm(item_repr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return item_repr / norms

    def recommend_similar_items(self, item_id_int, topK=10):
        vec = self.item_embeddings[item_id_int].reshape(1, -1)
        sims = cosine_similarity(vec, self.item_embeddings).flatten()
        sims[item_id_int] = -np.inf
        top_idx = np.argsort(-sims)[:topK]
        top_scores = sims[top_idx]
        top_mahang = [self.item_enc.inverse_transform([i])[0] for i in top_idx]
        return list(zip(top_idx.tolist(), top_mahang, top_scores.tolist()))
    
    def recommend_similar_items_by_mahang(self, ma_hang, topK=10) -> RecommendProductInfo:
        item_id_int = self.item_enc.transform([ma_hang])[0]
        recommendations = self.recommend_similar_items(item_id_int, topK)
        
        order_list = [
            OrderInfo(
                product_id=sim_mahang,
                score=float(score),
                rank=rank + 1
            )
            for rank, (_, sim_mahang, score) in enumerate(recommendations)
        ]

        return RecommendProductInfo(
            product_id=ma_hang,
            recommendations=order_list
        )
