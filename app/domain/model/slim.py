import numpy as np
import json
import pickle
import scipy.sparse as sp

class SLIM_Predictor:
    def __init__(self, W_matrix, item_map):
        self.W = W_matrix
        self.item_map = item_map
        self.inv_item_map = {v: k for k, v in item_map.items()}
        self.n_items = W_matrix.shape[0]

    def recommend(self, cart_items, top_k=5):
        # 1. Tạo vector User ảo từ giỏ hàng
        user_vector = np.zeros(self.n_items)
        valid_items = []
        
        for item_code in cart_items:
            if item_code in self.item_map:
                idx = self.item_map[item_code]
                if idx < self.n_items:
                    user_vector[idx] = 1.0
                    valid_items.append(item_code)
        
        if len(valid_items) == 0:
            return json.dumps({"status": "warning", "message": "Giỏ hàng rỗng", "recommendations": []}, ensure_ascii=False)

        # 2. Tính điểm (Score = Vector x Matrix W)
        if sp.issparse(self.W):
            scores = self.W.T.dot(user_vector) # Nhân vector với ma trận thưa
            # Lưu ý: SLIM thường là R * W, nhưng ở đây ta dùng vector * W. 
            # Nếu W là (Item x Item), ta có thể dùng .dot
        else:
            scores = np.dot(user_vector, self.W)
            
        # --- ĐOẠN FIX MẠNH TAY: ÉP KIỂU VỀ MẢNG 1 CHIỀU ---
        # Bất kể nó là sparse, matrix hay gì, đều ép về numpy array 1D
        if sp.issparse(scores):
            scores = scores.toarray() # Chuyển thưa -> đặc (2D)
        
        # Đảm bảo nó là numpy array (đề phòng nó là np.matrix)
        scores = np.asarray(scores)
        
        # Làm phẳng thành 1 chiều (1D array)
        scores = scores.ravel() 
        # --------------------------------------------------

        # 3. Lọc bỏ các món ĐANG CÓ trong giỏ (Gán -vô cực)
        for item_code in valid_items:
            idx = self.item_map[item_code]
            if idx < len(scores):
                scores[idx] = -np.inf 

        # 4. Chuẩn hóa điểm số
        # (Lúc này scores đã là numpy array 100%, so sánh thoải mái)
        valid_scores = scores[scores > -np.inf]
        if len(valid_scores) > 0:
            max_score = np.max(valid_scores)
            if max_score > 0:
                scores = (scores / max_score) * 100.0

        # 5. Lấy Top K
        top_indices = np.argsort(-scores)[:100]
        
        rec_list = []
        for idx in top_indices:
            if idx in self.inv_item_map:
                score_val = float(scores[idx])
                if score_val > 0.01:
                    rec_list.append({
                        "product_id": self.inv_item_map[idx],
                        "score": round(score_val, 2)
                    })
                if len(rec_list) == top_k:
                    break
            
        return json.dumps({
            "status": "success", 
            "input_cart": valid_items, 
            "recommendations": rec_list
        }, indent=4, ensure_ascii=False)

    def save(self, filename="slim_full_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói mô hình SLIM vào: {filename}")

    @staticmethod
    def load(filename="slim_full_model.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)