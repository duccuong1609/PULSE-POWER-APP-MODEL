import torch
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
from app.domain.dto.recommend_customer_info import RecommendInfo, OrderInfo

class RecommendationCustomerServiceNeuMF:
    def __init__(self, model, ckpt, user_enc: LabelEncoder, item_enc: LabelEncoder, train_user_pos: Dict[int, set]):
        self.model = model
        self.ckpt = ckpt
        self.user_enc = user_enc
        self.item_enc = item_enc
        self.train_user_pos = train_user_pos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def recommend_user_topk(self, user_id_int: int, topK: int = 10):
        """
        Recommend top-K items for a given user_id_int.
        Returns a RecommendInfo object with ordered recommendations.
        """
        self.model.eval()
        n_items = len(self.item_enc.classes_)

        with torch.no_grad():
            scores = np.empty(n_items, dtype=np.float32)
            chunk_size = 1024
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                users = torch.full((end - start,), user_id_int, dtype=torch.long, device=self.device)
                items = torch.arange(start, end, dtype=torch.long, device=self.device)
                logits = self.model(users, items)
                scores[start:end] = torch.sigmoid(logits).detach().cpu().numpy()

            seen_items = list(self.train_user_pos.get(user_id_int, []))
            if seen_items:
                scores[seen_items] = -np.inf

            top_items_idx = np.argsort(-scores)[:topK]
            top_item_codes = self.item_enc.inverse_transform(top_items_idx)

            recommendations = [
                OrderInfo(product_id=code, score=float(scores[idx]), rank=i + 1)
                for i, (idx, code) in enumerate(zip(top_items_idx, top_item_codes))
            ]

        return recommendations

    def recommend_user_topk_by_customer_id(self, ma_kh: str, topK: int = 10) -> RecommendInfo:
        """
        Wrapper cho phép truyền mã khách hàng gốc (MaKH)
        Trả về RecommendInfo với list[OrderInfo]
        """
        user_id_int = self.user_enc.transform([ma_kh])[0]
        recommendations = self.recommend_user_topk(user_id_int, topK)
        return RecommendInfo(user_id=str(ma_kh), recommendations=recommendations)