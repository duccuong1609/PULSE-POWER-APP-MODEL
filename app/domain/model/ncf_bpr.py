import torch
import torch.nn as nn

class NCF_BPR(nn.Module):
    def __init__(self, num_users, num_items, emb_size=128, hidden=[256,128,64], dropout=0.2, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Embedding
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        # MLP layers
        layers, input_size = [], emb_size*2
        for h in hidden:
            layers += [nn.Linear(input_size, h), nn.ReLU(), nn.Dropout(dropout)]
            input_size = h
        self.mlp = nn.Sequential(*layers)

        # Final prediction layer
        self.predict_layer = nn.Linear(input_size, 1)

        # Init embeddings
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.to(self.device)

    def forward(self, u, i):
        u = u.to(self.device)
        i = i.to(self.device)
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        x = torch.cat([u_e, i_e], dim=1)
        h = self.mlp(x)
        return self.predict_layer(h).squeeze(-1)

    def score(self, u_idx, i_idx):
        self.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([u_idx], device=self.device)
            i_tensor = torch.tensor([i_idx], device=self.device)
            return self.forward(u_tensor, i_tensor).item()