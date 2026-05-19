"""Linear encoder that fuses per-edge engineered features with node features.

Mirrors :class:`LinearEncoder` for the node-feature path, then projects the
edge feature tensor through two independent linears (src/dst) and adds them to
the indexed source/destination node embeddings. The result is per-edge
``h_src`` and ``h_dst`` tensors compatible with ``EdgeLinearDecoder`` and
``CustomEdgeMLP``.

The symmetric split (one linear for the src contribution, one for the dst
contribution) lets the decoder learn an asymmetric mapping of edge-feature
signal to the two endpoints, which is what we want when edge features describe
both the *source's* recent activity (fan-out, op rarity, burstiness) and the
*destination's* recent activity (fan-in) on the same vector.
"""

import torch
import torch.nn as nn


class LinearEdgeFeatEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin_x = nn.Linear(in_dim, out_dim)
        self.lin_e_src = nn.Linear(edge_dim, out_dim) if edge_dim and edge_dim > 0 else None
        self.lin_e_dst = nn.Linear(edge_dim, out_dim) if edge_dim and edge_dim > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, *, x, edge_index, edge_feats=None, **kwargs):
        # Mirror LinearEncoder's tuple/list handling so this drops in cleanly.
        if isinstance(x, (tuple, list)):
            h_src_nodes = self.dropout(self.lin_x(x[0]))
            h_dst_nodes = self.dropout(self.lin_x(x[1]))
        else:
            h = self.dropout(self.lin_x(x))
            h_src_nodes = h_dst_nodes = h

        # Per-edge embeddings via gather.
        h_src = h_src_nodes[edge_index[0]]
        h_dst = h_dst_nodes[edge_index[1]]

        if edge_feats is not None and self.lin_e_src is not None:
            e = self.dropout(edge_feats)
            h_src = h_src + self.lin_e_src(e)
            h_dst = h_dst + self.lin_e_dst(e)

        # Return both node-level h (for code paths that index by edge_index in
        # gather_h) and the pre-indexed per-edge h_src/h_dst (which gather_h
        # will prefer when present).
        return {"h": h_src_nodes, "h_src": h_src, "h_dst": h_dst}
