#!/usr/bin/env python3
"""Plot basic distribution views for ARGUS node embeddings.

Outputs:
- l2_norm_histogram.png
- pca_scatter.png
- umap_scatter.png (if umap-learn is installed)
- embedding_stats.txt
"""

import argparse
import os
import random

import numpy as np
import torch


def _sample_embeddings(embeddings_dict, sample_size, seed):
    node_ids = list(embeddings_dict.keys())
    total = len(node_ids)
    if total == 0:
        raise ValueError("Embedding dictionary is empty.")

    random.seed(seed)
    if sample_size <= 0 or sample_size >= total:
        selected_ids = node_ids
    else:
        selected_ids = random.sample(node_ids, sample_size)

    matrix = np.vstack([np.asarray(embeddings_dict[nid], dtype=np.float32) for nid in selected_ids])
    return selected_ids, matrix


def _save_stats(matrix, out_dir):
    norms = np.linalg.norm(matrix, axis=1)
    stats_path = os.path.join(out_dir, "embedding_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"num_embeddings={matrix.shape[0]}\n")
        f.write(f"embedding_dim={matrix.shape[1]}\n")
        f.write(f"l2_norm_mean={norms.mean():.6f}\n")
        f.write(f"l2_norm_std={norms.std():.6f}\n")
        f.write(f"l2_norm_min={norms.min():.6f}\n")
        f.write(f"l2_norm_max={norms.max():.6f}\n")
        f.write(f"l2_norm_p01={np.percentile(norms, 1):.6f}\n")
        f.write(f"l2_norm_p50={np.percentile(norms, 50):.6f}\n")
        f.write(f"l2_norm_p99={np.percentile(norms, 99):.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot distribution of ARGUS embeddings")
    parser.add_argument("embeddings_path", help="Path to bert_node_embeddings.pt")
    parser.add_argument(
        "--out_dir",
        default="artifacts/analysis/argus_embedding_plots",
        help="Directory where plots and stats are saved",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of embeddings to sample for 2D plots; <=0 uses all",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--bins", type=int, default=80, help="Bins for L2 norm histogram")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    embeddings = torch.load(args.embeddings_path, map_location="cpu")
    if not isinstance(embeddings, dict):
        raise TypeError(f"Expected dict[node_id -> embedding], got {type(embeddings)}")

    _, matrix = _sample_embeddings(embeddings, args.sample_size, args.seed)
    norms = np.linalg.norm(matrix, axis=1)

    # Matplotlib is only imported at runtime to keep import errors clear.
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 1) L2 norm histogram
    plt.figure(figsize=(8, 5))
    plt.hist(norms, bins=args.bins, edgecolor="black", linewidth=0.3)
    plt.title("ARGUS Embedding L2 Norm Distribution")
    plt.xlabel("L2 norm")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "l2_norm_histogram.png"), dpi=180)
    plt.close()

    # 2) PCA scatter (2D)
    pca = PCA(n_components=2, random_state=args.seed)
    xy = pca.fit_transform(matrix)

    plt.figure(figsize=(7, 7))
    plt.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.35)
    plt.title(
        f"PCA of ARGUS Embeddings (var={pca.explained_variance_ratio_.sum():.2%})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pca_scatter.png"), dpi=180)
    plt.close()

    # 3) UMAP scatter (optional)
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        umap_xy = reducer.fit_transform(matrix)

        plt.figure(figsize=(7, 7))
        plt.scatter(umap_xy[:, 0], umap_xy[:, 1], s=2, alpha=0.35)
        plt.title("UMAP of ARGUS Embeddings")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "umap_scatter.png"), dpi=180)
        plt.close()
        umap_status = "generated"
    except Exception as exc:
        umap_status = f"skipped ({exc})"

    _save_stats(matrix, args.out_dir)

    print(f"Saved plots and stats to: {os.path.abspath(args.out_dir)}")
    print(f"Used {matrix.shape[0]} embeddings with dim={matrix.shape[1]}")
    print(f"UMAP: {umap_status}")


if __name__ == "__main__":
    main()
