#!/usr/bin/env python3
"""Quick inspector for ARGUS BERT node embeddings (.pt)."""

import argparse
import random

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a few node embeddings from a .pt file")
    parser.add_argument("embeddings_path", help="Path to bert_node_embeddings.pt")
    parser.add_argument("--num_examples", type=int, default=3, help="How many examples to print")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample random node ids instead of taking the first ones",
    )
    args = parser.parse_args()

    embeddings = torch.load(args.embeddings_path, map_location="cpu")

    if not isinstance(embeddings, dict):
        raise TypeError(f"Expected dict[node_id -> embedding], got {type(embeddings)}")

    node_ids = list(embeddings.keys())
    total = len(node_ids)
    print(f"Loaded {total} node embeddings from: {args.embeddings_path}")

    if total == 0:
        print("Embedding dictionary is empty.")
        return

    num = max(1, min(args.num_examples, total))
    selected = random.sample(node_ids, num) if args.random else node_ids[:num]

    for i, node_id in enumerate(selected, start=1):
        vec = np.asarray(embeddings[node_id])
        head = np.array2string(vec[:8], precision=4, separator=", ")
        norm = float(np.linalg.norm(vec))

        print("-" * 72)
        print(f"Example {i}")
        print(f"node_id: {node_id}")
        print(f"shape: {vec.shape}")
        print(f"dtype: {vec.dtype}")
        print(f"L2 norm: {norm:.6f}")
        print(f"first 8 values: {head}")


if __name__ == "__main__":
    main()
