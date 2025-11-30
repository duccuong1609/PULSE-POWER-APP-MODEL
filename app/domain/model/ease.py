import numpy as np
import json
import pickle

class EASE_Predictor:
    def __init__(self, B_matrix, item_map):
        """
        Khởi tạo mô hình với Ma trận trọng số B và Item Map.
        """
        self.B = B_matrix
        self.item_map = item_map
        self.inv_item_map = {v: k for k, v in item_map.items()}
        self.n_items = B_matrix.shape[0]

    def recommend(self, cart_items, top_k=5):
        """
        Hàm dự đoán: Input là giỏ hàng -> Output là JSON gợi ý
        """
        user_vector = np.zeros(self.n_items)
        valid_items = []
        
        for item_code in cart_items:
            if item_code in self.item_map:
                idx = self.item_map[item_code]
                if idx < self.n_items:
                    user_vector[idx] = 1.0
                    valid_items.append(item_code)
        
        if len(valid_items) == 0:
            return json.dumps({
                "status": "warning",
                "message": "Giỏ hàng rỗng hoặc mã sản phẩm không tồn tại",
                "recommendations": []
            }, ensure_ascii=False)

        if hasattr(self.B, "dot"):
            scores = user_vector.dot(self.B)
        else:
            scores = np.dot(user_vector, self.B)
            
        for item_code in valid_items:
            idx = self.item_map[item_code]
            if idx < len(scores):
                scores[idx] = -np.inf 
        
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

    def save(self, filename="ease_full_model.pkl"):
        """Lưu toàn bộ đối tượng xuống file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói mô hình EASE vào: {filename}")

    @staticmethod
    def load(filename="ease_full_model.pkl"):
        """Load đối tượng lên từ file"""
        with open(filename, "rb") as f:
            return pickle.load(f)