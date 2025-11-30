import numpy as np
import json
import pickle
import scipy.sparse as sparse

class ItemKNN_Predictor:
    def __init__(self, sim_matrix, item_map):
        """
        Khởi tạo bộ dự đoán Item-kNN.
        - sim_matrix: Ma trận tương đồng (Item x Item).
        - item_map: Dictionary mapping mã SP -> Index.
        """
        self.sim_matrix = sim_matrix
        self.item_map = item_map
        self.inv_item_map = {v: k for k, v in item_map.items()}
        self.n_items = sim_matrix.shape[0]

    def recommend(self, cart_items, top_k=5):
        """
        Hàm gợi ý từ giỏ hàng. Trả về Raw Score.
        """
        # 1. Tạo vector User ảo từ giỏ hàng
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

        # 2. Tính điểm (Inference)
        # Công thức: Score = User_Vector x Similarity_Matrix
        # (Nghĩa là: Cộng tổng độ tương đồng của các món trong giỏ với các món khác)
        if sparse.issparse(self.sim_matrix):
            scores = user_vector.dot(self.sim_matrix)
        else:
            scores = np.dot(user_vector, self.sim_matrix)
            
        # 3. Lọc bỏ các món ĐANG CÓ trong giỏ (Gán -vô cực)
        for item_code in valid_items:
            idx = self.item_map[item_code]
            if idx < len(scores):
                scores[idx] = -np.inf 

        # 4. Lấy Top K (Lấy dư 100 để lọc index ảo)
        top_indices = np.argsort(-scores)[:100]
        
        rec_list = []
        for idx in top_indices:
            if idx in self.inv_item_map:
                score_val = float(scores[idx])
                
                # Chỉ lấy điểm dương (Cosine Similarity > 0)
                if score_val > 0.0001:
                    rec_list.append({
                        "product_id": self.inv_item_map[idx],
                        "score": round(score_val, 4) # Giữ nguyên giá trị thô, làm tròn 4 số
                    })
                
                if len(rec_list) == top_k:
                    break
            
        return json.dumps({
            "status": "success", 
            "input_cart": valid_items, 
            "recommendations": rec_list
        }, indent=4, ensure_ascii=False)

    def save(self, filename="knn_full_model.pkl"):
        """Lưu trọn gói xuống file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu trọn gói mô hình Item-kNN vào: {filename}")

    @staticmethod
    def load(filename="knn_full_model.pkl"):
        """Load lên dùng"""
        with open(filename, "rb") as f:
            return pickle.load(f)