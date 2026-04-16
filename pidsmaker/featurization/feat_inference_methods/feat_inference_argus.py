import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from pidsmaker.featurization.featurization_methods.featurization_argus import get_node2corpus
from pidsmaker.utils.utils import log, log_start, log_tqdm


def infer_pooled(document, encoder_model, tokenizer, max_length, device, output_dim):
    """Infer pooled (non-projected) CodeBERT embedding from a neighborhood text."""
    if not document:
        return np.zeros(output_dim)

    sentence = " ".join(document)
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)
        mask = inputs["attention_mask"].squeeze(0).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)

    return pooled.detach().cpu().numpy()


class LinearProjectionAE(nn.Module):
    """Trainable linear projection head (encoder) with reconstruction decoder."""

    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, proj_dim)
        self.decoder = nn.Linear(proj_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def train_projection_head(pooled_vectors, proj_dim, device, epochs, batch_size, learning_rate, seed):
    """Train linear projection (CodeBERT frozen) to compress pooled vectors."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.from_numpy(pooled_vectors).to(torch.float32)
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LinearProjectionAE(in_dim=x.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            _, recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        mean_loss = epoch_loss / max(1, batch_count)
        log(f"ARGUS projection training [{epoch + 1}/{epochs}] - recon_loss={mean_loss:.6f}")

    return model.encoder.eval()


def main(cfg):
    """Infer ARGUS node embeddings from adapted or base CodeBERT."""
    log_start(__file__)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning,
    )

    argus_cfg = cfg.featurization.argus
    adapt_codebert = bool(getattr(argus_cfg, "adapt_codebert", True))
    model_name = getattr(argus_cfg, "codebert_model_name", "microsoft/codebert-base")

    trained_model_dir = os.path.join(cfg.featurization._model_dir, "codebert_mlm_final")
    required_files = [
        os.path.join(trained_model_dir, "config.json"),
        os.path.join(trained_model_dir, "tokenizer_config.json"),
    ]
    has_local_adapted_model = os.path.isdir(trained_model_dir) and all(
        os.path.exists(p) for p in required_files
    )

    if adapt_codebert:
        if not has_local_adapted_model:
            raise FileNotFoundError(
                "Missing adapted CodeBERT artifacts for argus inference. "
                f"Expected folder/files under: {trained_model_dir}. "
                "Run featurization first (or force restart) so domain-adapted MLM weights are saved, "
                "e.g. --force_restart featurization,feat_inference. "
                "Or set argus.adapt_codebert=False to use the base model directly."
            )
        model_source = trained_model_dir
        log(f"Loading adapted CodeBERT from {model_source}...")
    else:
        model_source = trained_model_dir if has_local_adapted_model else model_name
        if has_local_adapted_model:
            log(
                "argus.adapt_codebert=False but found local adapted model; "
                f"using local model at {model_source}."
            )
        else:
            log(
                "argus.adapt_codebert=False; using base pretrained CodeBERT "
                f"model: {model_source}."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    encoder_model = AutoModel.from_pretrained(model_source)

    use_cpu = getattr(cfg, "_use_cpu", False)
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    max_length = int(getattr(argus_cfg, "max_length", 512))
    output_dim = int(cfg.featurization.emb_dim)
    if output_dim <= 0:
        raise ValueError(f"Invalid featurization.emb_dim={output_dim}. Expected a positive integer.")

    hidden_size = int(encoder_model.config.hidden_size)
    expected_size_with_avg = hidden_size + 1

    node2corpus, node2avg_interaction_count = get_node2corpus(
        cfg,
        splits=["train", "val", "test"],
        return_avg_interaction_count=True,
    )
    log(f"Embedding {len(node2corpus)} nodes with ARGUS...")
    node_items = list(node2corpus.items())
    total_nodes = len(node_items)
    indexid2vec = {}

    if output_dim == 1:
        log("emb_dim=1: using only avg_interaction_count as node feature.")
        for indexid, _ in log_tqdm(
            node_items,
            desc="Building avg-only node embeddings",
            total=total_nodes,
            disable=False,
        ):
            avg_interaction_count = float(node2avg_interaction_count.get(indexid, 0.0))
            indexid2vec[indexid] = np.array([avg_interaction_count], dtype=np.float32)
        return indexid2vec

    if output_dim < expected_size_with_avg:
        proj_dim = output_dim - 1
        projection_epochs = max(1, int(getattr(argus_cfg, "projection_train_epochs", 3)))
        projection_batch_size = max(1, int(getattr(argus_cfg, "projection_batch_size", 512)))
        projection_lr = float(getattr(argus_cfg, "projection_learning_rate", 0.001))
        seed = int(getattr(cfg.featurization, "seed", 0))

        log(
            "CodeBERT+avg dimension "
            f"({expected_size_with_avg}) exceeds emb_dim ({output_dim}); "
            f"training a frozen-CodeBERT projection 768->{proj_dim} then appending avg."
        )

        pooled_vectors = np.zeros((total_nodes, hidden_size), dtype=np.float32)
        avg_values = np.zeros((total_nodes,), dtype=np.float32)

        for i, (indexid, corpus) in enumerate(
            log_tqdm(
                node_items,
                desc="Computing pooled CodeBERT embeddings",
                total=total_nodes,
                disable=False,
            )
        ):
            pooled_vectors[i] = infer_pooled(
                corpus,
                encoder_model,
                tokenizer,
                max_length=max_length,
                device=device,
                output_dim=hidden_size,
            )
            avg_values[i] = float(node2avg_interaction_count.get(indexid, 0.0))

        projection = train_projection_head(
            pooled_vectors=pooled_vectors,
            proj_dim=proj_dim,
            device=device,
            epochs=projection_epochs,
            batch_size=projection_batch_size,
            learning_rate=projection_lr,
            seed=seed,
        )

        os.makedirs(cfg.feat_inference._model_dir, exist_ok=True)
        projection_path = os.path.join(cfg.feat_inference._model_dir, "argus_projection_encoder.pt")
        torch.save(
            {
                "state_dict": projection.state_dict(),
                "in_dim": hidden_size,
                "out_dim": proj_dim,
            },
            projection_path,
        )
        log(f"Saved ARGUS projection encoder to {projection_path}")

        projection.eval()
        with torch.no_grad():
            x = torch.from_numpy(pooled_vectors).to(torch.float32)
            projected = np.zeros((total_nodes, proj_dim), dtype=np.float32)
            for start in log_tqdm(
                range(0, total_nodes, projection_batch_size),
                desc="Applying projection to node embeddings",
                total=(total_nodes + projection_batch_size - 1) // projection_batch_size,
                disable=False,
            ):
                stop = min(start + projection_batch_size, total_nodes)
                projected_batch = projection(x[start:stop].to(device)).cpu().numpy()
                projected[start:stop] = projected_batch

        for i, (indexid, _) in enumerate(node_items):
            vector = np.concatenate([projected[i], np.array([avg_values[i]], dtype=np.float32)])
            indexid2vec[indexid] = vector

    else:
        if expected_size_with_avg != output_dim:
            log(
                "Warning: CodeBERT + avg interaction feature size "
                f"({hidden_size} + 1 = {expected_size_with_avg}) differs from emb_dim ({output_dim}). "
                "Vectors will be padded to match emb_dim."
            )

        for indexid, corpus in log_tqdm(
            node_items,
            desc="Embedding all nodes in the dataset",
            total=total_nodes,
            disable=False,
        ):
            avg_interaction_count = float(node2avg_interaction_count.get(indexid, 0.0))
            vector = infer_pooled(
                corpus,
                encoder_model,
                tokenizer,
                max_length=max_length,
                device=device,
                output_dim=hidden_size,
            )
            vector = np.concatenate(
                [vector, np.array([avg_interaction_count], dtype=vector.dtype)],
                axis=0,
            )
            if len(vector) < output_dim:
                vector = np.pad(vector, (0, output_dim - len(vector)), mode="constant")
            indexid2vec[indexid] = vector

    return indexid2vec
