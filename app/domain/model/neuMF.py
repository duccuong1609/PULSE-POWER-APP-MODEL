import torch
import torch.nn as nn

EMBED_DIM_GMF = 64
EMBED_DIM_MLP = 64
MLP_LAYER_SIZES = [128, 64, 32]
DROPOUT = 0.1

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim_gmf=EMBED_DIM_GMF, emb_dim_mlp=EMBED_DIM_MLP, mlp_layers=MLP_LAYER_SIZES, dropout=DROPOUT):
        super().__init__()
        # embeddings
        self.user_embed_gmf = nn.Embedding(n_users, emb_dim_gmf)
        self.item_embed_gmf = nn.Embedding(n_items, emb_dim_gmf)
        self.user_embed_mlp = nn.Embedding(n_users, emb_dim_mlp)
        self.item_embed_mlp = nn.Embedding(n_items, emb_dim_mlp)
        # MLP
        mlp_input = emb_dim_mlp * 2
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_input, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_input = h
        self.mlp = nn.Sequential(*layers)
        # final fusion
        final_input = emb_dim_gmf + (mlp_layers[-1] if len(mlp_layers)>0 else emb_dim_mlp)
        self.predict = nn.Linear(final_input, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.predict.weight)

    def forward(self, user_indices, item_indices):
        ug_gmf = self.user_embed_gmf(user_indices)
        ig_gmf = self.item_embed_gmf(item_indices)
        gmf_out = ug_gmf * ig_gmf

        ug_mlp = self.user_embed_mlp(user_indices)
        ig_mlp = self.item_embed_mlp(item_indices)
        mlp_in = torch.cat([ug_mlp, ig_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        neu = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.predict(neu).squeeze(-1)
        return logits