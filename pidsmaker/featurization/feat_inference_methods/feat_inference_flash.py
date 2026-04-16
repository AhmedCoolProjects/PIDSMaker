import os
import warnings
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pidsmaker.featurization.featurization_methods.featurization_flash import get_node2corpus
from pidsmaker.utils.utils import log, log_start, log_tqdm


def infer(document, encoder_model, tokenizer, max_length, device, output_dim):
    """
    Each node is associated to a `document` which is the list of (msg => edge type => msg)
    involving this node.
    We get the embedding of each word inside this document and we do the mean of all embeddings.
    OOV words are simply ignored.
    """
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
    if len(vector) > output_dim:
        return vector[:output_dim]
    if len(vector) < output_dim:
        return np.pad(vector, (0, output_dim - len(vector)), mode="constant")
    return vector


def main(cfg):
    log_start(__file__)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning,
    )

    flash_cfg = cfg.featurization.flash
    adapt_codebert = bool(getattr(flash_cfg, "adapt_codebert", True))
    model_name = getattr(flash_cfg, "codebert_model_name", "microsoft/codebert-base")

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
                "Missing adapted CodeBERT artifacts for flash inference. "
                f"Expected folder/files under: {trained_model_dir}. "
                "Run featurization first (or force restart) so domain-adapted MLM weights are saved, "
                "e.g. --force_restart featurization,feat_inference. "
                "Or set flash.adapt_codebert=False to use the base model directly."
            )
        model_source = trained_model_dir
        log(f"Loading adapted CodeBERT from {model_source}...")
    else:
        model_source = trained_model_dir if has_local_adapted_model else model_name
        if has_local_adapted_model:
            log(
                "flash.adapt_codebert=False but found local adapted model; "
                f"using local model at {model_source}."
            )
        else:
            log(
                "flash.adapt_codebert=False; using base pretrained CodeBERT "
                f"model: {model_source}."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    encoder_model = AutoModel.from_pretrained(model_source)

    use_cpu = getattr(cfg, "_use_cpu", False)
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    max_length = int(getattr(flash_cfg, "max_length", 512))
    output_dim = int(cfg.featurization.emb_dim)
    progress_log_every = int(getattr(flash_cfg, "progress_log_every", 1000))

    hidden_size = int(encoder_model.config.hidden_size)
    if hidden_size != output_dim:
        log(
            f"Warning: CodeBERT hidden size ({hidden_size}) differs from emb_dim ({output_dim}). "
            "Vectors will be truncated/padded to match emb_dim."
        )

    node2corpus = get_node2corpus(cfg, splits=["train", "val", "test"])
    log(f"Embedding {len(node2corpus)} nodes with FLASH...")
    indexid2vec = {}
    total_nodes = len(node2corpus)
    for i, (indexid, corpus) in enumerate(
        log_tqdm(
        node2corpus.items(),
        desc="Embedding all nodes in the dataset",
        total=total_nodes,
    ),
        start=1,
    ):
        indexid2vec[indexid] = infer(
            corpus,
            encoder_model,
            tokenizer,
            max_length=max_length,
            device=device,
            output_dim=output_dim,
        )

        if progress_log_every > 0 and (i % progress_log_every == 0 or i == total_nodes):
            log(f"Embedded {i}/{total_nodes} nodes ({(100.0 * i / total_nodes):.1f}%).")

    return indexid2vec
