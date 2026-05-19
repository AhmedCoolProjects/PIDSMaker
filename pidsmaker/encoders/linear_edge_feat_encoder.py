import torch
import torch.nn as nn


class LinearEdgeFeatEncoder(nn.Module):
    """Edge-aware linear encoder for VELOX-style per-edge classification.

    Encodes source and destination node features, then fuses edge features into
    both branches so the downstream edge decoder can leverage behavioral context.
    """

    def __init__(
        self,
        in_dim,
        edge_dim,
        out_dim,
        dropout=0.0,
        use_gating=True,
        use_layer_norm=True,
    ):
        super().__init__()
        self.use_gating = bool(use_gating)
        self.use_layer_norm = bool(use_layer_norm)
        self.edge_dim = int(edge_dim or 0)

        self.src_lin = nn.Linear(in_dim, out_dim)
        self.dst_lin = nn.Linear(in_dim, out_dim)

        if self.edge_dim > 0:
            self.edge_lin = nn.Linear(self.edge_dim, out_dim)
            if self.use_gating:
                self.gate_lin = nn.Linear(out_dim * 3, out_dim)

        self.src_norm = nn.LayerNorm(out_dim) if self.use_layer_norm else nn.Identity()
        self.dst_norm = nn.LayerNorm(out_dim) if self.use_layer_norm else nn.Identity()
        self.edge_norm = (
            nn.LayerNorm(out_dim) if self.use_layer_norm and self.edge_dim > 0 else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def _resolve_src_dst(self, x, x_src, x_dst):
        if x_src is not None and x_dst is not None:
            return x_src, x_dst
        if isinstance(x, (tuple, list)):
            return x[0], x[1]
        return x, x

    def forward(self, x=None, x_src=None, x_dst=None, edge_feats=None, **kwargs):
        x_src, x_dst = self._resolve_src_dst(x=x, x_src=x_src, x_dst=x_dst)

        src_h = self.src_norm(self.src_lin(x_src))
        dst_h = self.dst_norm(self.dst_lin(x_dst))

        if self.edge_dim > 0 and edge_feats is not None:
            edge_h = self.edge_norm(self.edge_lin(edge_feats))
            if self.use_gating:
                gate = torch.sigmoid(self.gate_lin(torch.cat([src_h, dst_h, edge_h], dim=-1)))
                edge_h = edge_h * gate

            h_src = self.dropout(src_h + edge_h)
            h_dst = self.dropout(dst_h + edge_h)
        else:
            h_src = self.dropout(src_h)
            h_dst = self.dropout(dst_h)

        return {
            "h": (h_src, h_dst),
            "h_src": h_src,
            "h_dst": h_dst,
        }
