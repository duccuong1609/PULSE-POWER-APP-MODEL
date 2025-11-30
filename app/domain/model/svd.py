import numpy as np
import pickle
import json

class SVD_Predictor:
    def __init__(self, item_factors, item_biases, raw_to_inner_id, inner_to_raw_id):
        self.qi = item_factors
        self.bi = item_biases
        self.raw_to_inner = raw_to_inner_id
        self.inner_to_raw = inner_to_raw_id
        self.n_items = item_factors.shape[0]

    def recommend(self, cart_items, top_k=5):
        # 1. Chuyển đổi mã SP
        valid_inner_ids = []
        valid_raw_ids = []
        
        for item in cart_items:
            item = str(item).strip().upper()
            if item in self.raw_to_inner:
                inner_id = self.raw_to_inner[item]
                valid_inner_ids.append(inner_id)
                valid_raw_ids.append(item)
        
        if not valid_inner_ids:
            return json.dumps({
                "status": "warning",
                "message": "Giỏ hàng rỗng",
                "recommendations": []
            }, ensure_ascii=False)

        # 2. Tính Vector trung bình của Giỏ hàng
        cart_vectors = self.qi[valid_inner_ids] 
        user_vector = np.mean(cart_vectors, axis=0) 

        # 3. Tính điểm (Raw Score = Dot Product + Bias)
        # Điểm này tương đương với Rating dự đoán (ví dụ: 3.5, 4.2...)
        scores = np.dot(self.qi, user_vector) + self.bi

        # 4. Lọc bỏ món đã có (Gán -vô cực)
        scores[valid_inner_ids] = -np.inf

        # --- ĐÃ BỎ ĐOẠN SCALING 0-100 ---
        # Giữ nguyên giá trị scores gốc

        # 5. Lấy Top K
        top_indices = np.argsort(-scores)[:100]

        rec_list = []
        for idx in top_indices:
            raw_id = self.inner_to_raw.get(idx, None)
            if raw_id:
                score_val = float(scores[idx])
                
                # Chỉ lấy nếu điểm > 0 (tránh gợi ý rating âm nếu có)
                # Với SVD rating thường là 1-5, nên điều kiện này an toàn
                if score_val > 0: 
                    rec_list.append({
                        "product_id": raw_id,
                        "score": round(score_val, 4) # Làm tròn 4 số
                    })
                if len(rec_list) == top_k:
                    break
        
        return json.dumps({
            "status": "success",
            "input_cart": valid_raw_ids,
            "recommendations": rec_list
        }, indent=4, ensure_ascii=False)

    def save(self, filename="svd_full_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói SVD (Raw Score) vào: {filename}")

    @staticmethod
    def load(filename="svd_full_model.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)