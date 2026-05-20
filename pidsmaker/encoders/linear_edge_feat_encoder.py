import torch
import torch.nn as nn


class LinearEdgeFeatEncoder(nn.Module):
    """Linear encoder that integrates edge features (engineered/temporal/types)

    directly with node embeddings to output edge-aligned source and destination representation.
    """

    def __init__(self, node_in_dim, edge_in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin_node = nn.Linear(node_in_dim, out_dim)
        if edge_in_dim > 0:
            self.lin_edge = nn.Linear(edge_in_dim, out_dim)
        else:
            self.lin_edge = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_src, x_dst, edge_feats=None, **kwargs):
        """Forward pass of the encoder.

        Args:
            x_src: Source node features (E, node_in_dim)
            x_dst: Destination node features (E, node_in_dim)
            edge_feats: Edge features (E, edge_in_dim)
            **kwargs: Extra parameters ignored

        Returns:
            dict: Dictionary with 'h', 'h_src', and 'h_dst' keys
        """
        h_src_node = self.lin_node(x_src)
        h_dst_node = self.lin_node(x_dst)

        if self.lin_edge is not None and edge_feats is not None:
            h_edge = self.lin_edge(edge_feats)
        else:
            h_edge = 0.0

        h_src = self.dropout(h_src_node + h_edge)
        h_dst = self.dropout(h_dst_node + h_edge)

        # Return format compatible with gather_h in model.py
        return {
            "h": (h_src, h_dst),
            "h_src": h_src,
            "h_dst": h_dst,
        }
