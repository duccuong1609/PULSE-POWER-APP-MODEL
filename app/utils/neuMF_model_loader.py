import torch
import os
from app.domain.model.neuMF import NeuMF
from torch.serialization import safe_globals
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model_neuMF(SAVE_PATH="app/assets/neuMF.pth"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"❌ File not found: {SAVE_PATH}")

    with safe_globals([np.core.multiarray._reconstruct]):
        ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)

    n_users = ckpt.get("n_users", 1061)
    n_items = ckpt.get("n_items", 55)
    
    user_enc = LabelEncoder()
    user_enc.classes_ = ckpt["user_enc_classes"]
    item_enc = LabelEncoder()
    item_enc.classes_ = ckpt["item_enc_classes"]
    
    train_user_pos = ckpt.get("train_user_pos")

    model = NeuMF(
        n_users=n_users,
        n_items=n_items,
        emb_dim_gmf=ckpt.get('emb_dim_gmf', 64),
        emb_dim_mlp=ckpt.get('emb_dim_mlp', 64),
        mlp_layers=ckpt.get('mlp_layers', [128,64,32]),
        dropout=ckpt.get('dropout', 0.1)
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        f"✅ Loaded NeuMF model from {SAVE_PATH} "
        f"(n_users={n_users}, n_items={n_items}, "
        f"best_ndcg10={ckpt.get('best_ndcg10', '?')})"
    )

    return model, ckpt, user_enc, item_enc, train_user_pos