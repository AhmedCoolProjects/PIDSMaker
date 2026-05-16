import torch
import torch.nn as nn


class EdgeEngineeredEncoder(nn.Module):
    """
    Linear encoder that also consumes engineered edge features.

    Behaves like LinearEncoder when no edge features are provided (falls back
    to projecting (x_src, x_dst) independently). When edge_feats are available,
    projects them and combines with src/dst node embeddings to produce a single
    per-edge embedding.

    Input:
        x: tuple (x_src_nodes, x_dst_nodes) each (N, d) (after reindexing)
        edge_feats: (E, edge_feat_dim) tensor of engineered features or None
        edge_index: (2, E) edge index tensor

    Output dict:
        {"h": (h_src_nodes, h_dst_nodes)}  -- fallback (no edge_feats)
        {"h": per_edge_emb, "h_src": per_edge_emb, "h_dst": per_edge_emb}  -- with edge_feats
    """

    def __init__(self, in_dim, edge_feat_dim, out_dim, dropout=0.0):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if edge_feat_dim and edge_feat_dim > 0:
            self.edge_proj = nn.Linear(edge_feat_dim, out_dim)
            self.combine = nn.Linear(out_dim * 3, out_dim)
        else:
            self.edge_proj = None
            self.combine = None

    def forward(self, x, edge_feats=None, edge_index=None, **kwargs):
        if isinstance(x, (tuple, list)):
            h_src_nodes = self.dropout(self.node_proj(x[0]))
            h_dst_nodes = self.dropout(self.node_proj(x[1]))
        else:
            h_src_nodes = self.dropout(self.node_proj(x))
            h_dst_nodes = h_src_nodes

        if edge_feats is not None and self.edge_proj is not None and edge_index is not None:
            h_src = h_src_nodes[edge_index[0]]
            h_dst = h_dst_nodes[edge_index[1]]
            h_edge = self.dropout(self.edge_proj(edge_feats))
            combined = torch.cat([h_src, h_dst, h_edge], dim=-1)
            h = self.combine(combined)
            return {"h": h, "h_src": h, "h_dst": h}

        return {"h": (h_src_nodes, h_dst_nodes)}
