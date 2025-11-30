import numpy as np
import json
import pickle
import scipy.sparse as sp

class Hybrid_Predictor:
    def __init__(self, ease_model, knn_model, slim_model, weights=(0.5, 0.3, 0.2)):
        self.ease = ease_model
        self.knn = knn_model
        self.slim = slim_model
        self.weights = weights
        # Dùng chung item_map của EASE (vì các model này chung tập train)
        self.item_map = ease_model.item_map 
        self.inv_item_map = ease_model.inv_item_map
        self.n_items = len(self.item_map)

    def _get_raw_score(self, model, user_vector):
        """
        Hàm phụ: Tính điểm thô từ 1 model con.
        Đã fix lỗi 'AttributeError: numpy.ndarray object has no attribute toarray'
        """
        # Tự động tìm xem model này lưu trọng số ở biến tên gì (B, W, hay sim_matrix)
        matrix = getattr(model, 'B', getattr(model, 'sim_matrix', getattr(model, 'W', None)))
        
        if matrix is None:
            return np.zeros(self.n_items) # Fallback nếu model rỗng

        # Tính toán điểm số
        if sp.issparse(matrix):
            # Nếu là Sparse (SLIM, kNN)
            scores = matrix.T.dot(user_vector)
            
            # --- FIX LỖI Ở ĐÂY ---
            # Chỉ gọi .toarray() nếu kết quả VẪN LÀ sparse matrix
            if sp.issparse(scores):
                scores = scores.toarray()
        else:
            # Nếu là Dense (EASE)
            scores = np.dot(user_vector, matrix)
            
        # Đảm bảo luôn trả về Numpy Array 1 chiều phẳng
        return np.asarray(scores).ravel()

    def _normalize(self, scores):
        """Min-Max Scaling vector về [0, 1]"""
        # Xử lý các giá trị -inf (do bước trước lọc) để không ảnh hưởng min/max
        # Ta chỉ scale trên các giá trị hợp lệ (khác -inf)
        valid_mask = scores > -np.inf
        if not np.any(valid_mask):
            return scores # Nếu toàn là -inf thì trả về luôn
            
        valid_scores = scores[valid_mask]
        min_val = np.min(valid_scores)
        max_val = np.max(valid_scores)
        
        if max_val - min_val == 0:
            return scores
            
        # Chỉ scale những thằng hợp lệ
        scores[valid_mask] = (scores[valid_mask] - min_val) / (max_val - min_val)
        return scores

    def recommend(self, cart_items, top_k=5):
        # 1. Tạo vector User ảo
        user_vector = np.zeros(self.n_items)
        valid_items = []
        for item_code in cart_items:
            if item_code in self.item_map:
                idx = self.item_map[item_code]
                if idx < self.n_items:
                    user_vector[idx] = 1.0
                    valid_items.append(item_code)
        
        if len(valid_items) == 0:
            return json.dumps({"status": "warning", "message": "Empty cart", "recommendations": []}, ensure_ascii=False)

        # 2. Lấy điểm từ 3 model con (Raw Scores)
        s1 = self._get_raw_score(self.ease, user_vector)
        s2 = self._get_raw_score(self.knn, user_vector)
        s3 = self._get_raw_score(self.slim, user_vector)

        # 3. Lọc bỏ món đã có trong giỏ TRƯỚC KHI normalize
        # (Để tránh -inf làm lệch Min-Max Scaling)
        for item_code in valid_items:
            idx = self.item_map[item_code]
            if idx < len(s1): s1[idx] = -np.inf
            if idx < len(s2): s2[idx] = -np.inf
            if idx < len(s3): s3[idx] = -np.inf

        # 4. Chuẩn hóa từng thằng về [0, 1]
        s1_norm = self._normalize(s1)
        s2_norm = self._normalize(s2)
        s3_norm = self._normalize(s3)

        # 5. Trộn (Weighted Sum)
        w1, w2, w3 = self.weights
        # Cộng dồn điểm (Lưu ý: -inf + số = -inf, nên các món trong giỏ vẫn an toàn)
        final_scores = (w1 * s1_norm) + (w2 * s2_norm) + (w3 * s3_norm)

        # 6. Lấy Top K
        top_indices = np.argsort(-final_scores)[:100]
        
        rec_list = []
        for idx in top_indices:
            if idx in self.inv_item_map:
                score_val = float(final_scores[idx])
                # Chỉ lấy những món có điểm > 0 (tức là có liên quan)
                if score_val > 0.001: 
                    rec_list.append({
                        "product_id": self.inv_item_map[idx],
                        "score": round(score_val, 4) # Điểm này max là 1.0
                    })
                if len(rec_list) == top_k:
                    break
        
        return json.dumps({
            "status": "success", 
            "input_cart": valid_items, 
            "recommendations": rec_list
        }, indent=4, ensure_ascii=False)

    def save(self, filename="hybrid_full_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói Hybrid vào: {filename}")

    @staticmethod
    def load(filename="hybrid_full_model.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)