import torch
import torch.nn as nn

class LinearEdgeFeatEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_feat_dim, dropout=0.0):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.edge_feat_proj = nn.Linear(edge_feat_dim, out_dim)
        self.fuse = nn.Linear(out_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x_src=None, x_dst=None, x=None, batch=None, *args, **kwargs):
        if x is not None and isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
            
        h_src = self.dropout(self.node_proj(x_src))
        h_dst = self.dropout(self.node_proj(x_dst))
        
        engineered_feats = getattr(batch, "engineered_feats", None) if batch is not None else None
        
        if engineered_feats is not None:
            h_edge = self.dropout(self.edge_feat_proj(engineered_feats))
            h_src = self.act(self.fuse(torch.cat([h_src, h_edge], dim=-1)))
            h_dst = self.act(self.fuse(torch.cat([h_dst, h_edge], dim=-1)))
            
        return {"h": (h_src, h_dst)}
