import os
import warnings

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pidsmaker.featurization.featurization_methods.featurization_argus import get_node2corpus
from pidsmaker.utils.utils import log, log_start, log_tqdm


def infer(document, encoder_model, tokenizer, max_length, device, output_dim, avg_interaction_count=0.0):
    """Infer a single ARGUS node embedding from tokenized neighborhood text."""
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

    vector = pooled.detach().cpu().numpy()
    vector = np.concatenate(
        [vector, np.array([float(avg_interaction_count)], dtype=vector.dtype)], axis=0
    )
    if len(vector) > output_dim:
        return vector[:output_dim]
    if len(vector) < output_dim:
        return np.pad(vector, (0, output_dim - len(vector)), mode="constant")
    return vector


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

    hidden_size = int(encoder_model.config.hidden_size)
    expected_size_with_avg = hidden_size + 1
    if expected_size_with_avg != output_dim:
        log(
            "Warning: CodeBERT + avg interaction feature size "
            f"({hidden_size} + 1 = {expected_size_with_avg}) differs from emb_dim ({output_dim}). "
            "Vectors will be truncated/padded to match emb_dim."
        )

    node2corpus, node2avg_interaction_count = get_node2corpus(
        cfg,
        splits=["train", "val", "test"],
        return_avg_interaction_count=True,
    )
    log(f"Embedding {len(node2corpus)} nodes with ARGUS...")
    indexid2vec = {}
    total_nodes = len(node2corpus)
    for indexid, corpus in log_tqdm(
        node2corpus.items(),
        desc="Embedding all nodes in the dataset",
        total=total_nodes,
        disable=False,
    ):
        avg_interaction_count = float(node2avg_interaction_count.get(indexid, 0.0))
        indexid2vec[indexid] = infer(
            corpus,
            encoder_model,
            tokenizer,
            max_length=max_length,
            device=device,
            output_dim=output_dim,
            avg_interaction_count=avg_interaction_count,
        )

    return indexid2vec
