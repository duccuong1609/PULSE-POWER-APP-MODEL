import torch
import numpy as np

class RecommendationCustomerService:
    def __init__(self, model, user2idx, item2idx, idx2user, idx2item, device):
        self.model = model
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.idx2user = idx2user
        self.idx2item = idx2item
        self.device = device
        self.num_items = len(item2idx)

    def recommend_top_k(self, user_id: str, top_k: int = 10):
        """
        Dự đoán top-K sản phẩm cho khách hàng.
        """
        if user_id not in self.user2idx:
            return []

        u_idx = self.user2idx[user_id]
        self.model.eval()

        users = torch.tensor([u_idx] * self.num_items, device=self.device)
        items = torch.tensor(list(range(self.num_items)), device=self.device)

        with torch.no_grad():
            scores = self.model(users, items).cpu().numpy()

        top_idx = scores.argsort()[::-1][:top_k]
        top_scores = scores[top_idx]

        if len(top_scores) > 1:
            top_scores_norm = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())
        else:
            top_scores_norm = np.array([1.0])

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append({
                "MaHang": self.idx2item[idx],
                "Score": float(top_scores_norm[rank-1]),
                "Rank": rank
            })

        return results