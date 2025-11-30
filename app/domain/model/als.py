import numpy as np
import json
import pickle

class ALS_Predictor:
    def __init__(self, model_input, item_map):
        """
        Khởi tạo bộ dự đoán ALS.
        Input: 
            - model_input: Có thể là model 'implicit' vừa train xong HOẶC ma trận numpy item_factors.
            - item_map: Dictionary mapping mã sản phẩm.
        """
        # 1. Tự động xử lý đầu vào (Model hay Array)
        if hasattr(model_input, "item_factors"):
            # Nếu là object của thư viện implicit
            self.item_factors = model_input.item_factors
        else:
            # Nếu là numpy array thô
            self.item_factors = model_input

        # 2. Đảm bảo chuyển về Numpy Array (để tránh lỗi GPU/CPU tensor)
        if not isinstance(self.item_factors, np.ndarray):
            self.item_factors = self.item_factors.to_numpy()

        self.item_map = item_map
        self.inv_item_map = {v: k for k, v in item_map.items()}
        self.n_items = self.item_factors.shape[0]

    def recommend(self, cart_items, top_k=5):
        """
        Hàm gợi ý sản phẩm từ giỏ hàng.
        """
        # 1. Tạo vector User ảo
        user_vector = np.zeros(self.n_items)
        valid_items = []
        
        for item_code in cart_items:
            if item_code in self.item_map:
                idx = self.item_map[item_code]
                if idx < self.n_items:
                    user_vector[idx] = 1.0
                    valid_items.append(item_code)
        
        # Xử lý giỏ hàng rỗng
        if len(valid_items) == 0:
            return json.dumps({
                "status": "warning",
                "message": "Giỏ hàng rỗng hoặc mã sản phẩm không tồn tại",
                "recommendations": []
            }, ensure_ascii=False)

        # 2. Tính điểm (Inference 2 bước để tiết kiệm RAM)
        # Bước A: User Latent = User x Factors
        user_latent = user_vector.dot(self.item_factors) 
        # Bước B: Scores = User Latent x Factors.T
        scores = user_latent.dot(self.item_factors.T)

        # 3. Lọc bỏ món đã có trong giỏ
        for item_code in valid_items:
            idx = self.item_map[item_code]
            if idx < len(scores):
                scores[idx] = -np.inf 

        # 5. Lấy Top K (Lấy dư 100 để lọc index ảo)
        top_indices = np.argsort(-scores)[:100] 
        
        rec_list = []
        for idx in top_indices:
            if idx in self.inv_item_map:
                score_val = float(scores[idx])
                
                # Chỉ lấy điểm dương > 0.01
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

    def save(self, filename="als_full_model.pkl"):
        """Lưu trọn gói đối tượng xuống file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói mô hình ALS vào: {filename}")

    @staticmethod
    def load(filename="als_full_model.pkl"):
        """Load đối tượng lên"""
        with open(filename, "rb") as f:
            return pickle.load(f)