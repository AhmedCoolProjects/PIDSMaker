import torch
import torch.nn as nn


class LinearEdgeFeatEncoder(nn.Module):
    """Linear encoder that fuses per-edge features into node embeddings.

    Projects node features ``x`` and edge features ``edge_feats`` to the same
    ``out_dim`` independently, then scatters the per-edge projection into both
    source and destination nodes via ``index_add``.  The result is added to the
    node-feature projection so downstream decoder sees edge-contextualised
    node embeddings.

    When ``edge_feats`` is ``None`` (e.g. backward-compat mode) this behaves
    identically to ``LinearEncoder``.
    """

    def __init__(self, in_dim: int, edge_feat_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_x = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(edge_feat_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x, edge_feats=None, edge_index=None, *args, **kwargs):
        h_x = self.dropout(self.lin_x(x))

        if edge_feats is not None and edge_index is not None:
            h_edge = self.dropout(self.lin_edge(edge_feats))
            h_edge_nodes = torch.zeros_like(h_x)
            h_edge_nodes = h_edge_nodes.index_add(0, edge_index[0], h_edge)
            h_edge_nodes = h_edge_nodes.index_add(0, edge_index[1], h_edge)
            h = h_x + h_edge_nodes
        else:
            h = h_x

        return {"h": h}
