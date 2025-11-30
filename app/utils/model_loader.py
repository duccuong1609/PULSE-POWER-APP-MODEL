import pickle
import os

class ModelLoader:
    _models = {}
    
    @classmethod
    def load_models(cls):
        """Load toàn bộ model từ file .pkl vào RAM"""
        print("⏳ Đang tải 6 mô hình vào bộ nhớ...")
        base_path = "assets"
        
        models_config = {
            "ease": "ease_full_model.pkl",
            "als": "als_full_model.pkl",
            "knn": "knn_full_model.pkl",
            "slim": "slim_full_model.pkl",
            "svd": "svd_full_model.pkl",
            "hybrid": "hybrid_full_5E_3K_2S_model.pkl"
        }

        for name, filename in models_config.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        cls._models[name] = pickle.load(f)
                    print(f"✅ Đã load xong: {name.upper()}")
                except Exception as e:
                    print(f"❌ Lỗi khi load {name}: {e}")
            else:
                print(f"⚠️ Không tìm thấy file: {filepath}")

    @classmethod
    def get_model(cls, model_name: str):
        """Lấy model theo tên"""
        return cls._models.get(model_name.lower())

model_loader = ModelLoader()