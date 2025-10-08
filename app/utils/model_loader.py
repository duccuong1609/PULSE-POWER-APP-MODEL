import torch
import joblib
from app.domain.model.ncf_bpr import NCF_BPR

def load_model_and_mappings(
    mapping_path="app/assets/ncf_bpr_user_mappings.pkl",
    model_path="app/assets/ncf_bpr_user_model.pth",
    emb_size=128,
    hidden=[256,128,64],
    dropout=0.2
):
    # ==== device ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Load mappings ====
    mappings = joblib.load(mapping_path)
    user2idx = mappings["user2idx"]
    item2idx = mappings["item2idx"]
    idx2user = mappings["idx2user"]
    idx2item = mappings["idx2item"]

    num_users = len(user2idx)
    num_items = len(item2idx)

    # ==== Khởi tạo model ====
    model = NCF_BPR(
        num_users=num_users,
        num_items=num_items,
        emb_size=emb_size,
        hidden=hidden,
        dropout=dropout,
        device=device
    )

    # ==== Load weights ====
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ NCF Model and mappings loaded successfully on {device}.")

    return model, user2idx, item2idx, idx2user, idx2item, device