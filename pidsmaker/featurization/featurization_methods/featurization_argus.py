import os
import random
import warnings
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_indexid2msg,
    log,
    log_start,
    log_tqdm,
)


def get_node2corpus(cfg, splits):
    indexid2msg = get_indexid2msg(cfg)

    def normalize_label_props(label: str) -> str:
        parts = [part for part in label.split() if part]
        normalized = [part for part in parts if part != "None"]
        return " ".join(normalized)

    def tokenize_flash_sentence(sentence: str):
        # Keep path/netflow-like entities as single tokens by splitting only on whitespace.
        # This avoids exploding sequence length when labels contain many '/' or '.' characters.
        return [tok for tok in sentence.split() if tok]

    split_to_paths = get_split_to_graph_paths(cfg, splits)
    sorted_paths = list(chain.from_iterable(split_to_paths.values()))

    data_of_graphs = []

    for file_path in log_tqdm(sorted_paths, desc=f"Loading training data for {str(splits)}"):
        graph = torch.load(file_path)

        sorted_edges = sorted(
            [
                (u, v, attr["label"], int(attr["time"]))
                for u, v, key, attr in graph.edges(data=True, keys=True)
            ],
            key=lambda x: x[3],
        )

        nodes, node_types = defaultdict(list), {}
        node_neighbor_events = defaultdict(list)
        for e in sorted_edges:
            src, dst, operation, t = e
            src_type, src_msg = indexid2msg[src]
            dst_type, dst_msg = indexid2msg[dst]

            node_types[src] = src_type
            node_types[dst] = dst_type

            # Keep 1-hop neighbor events for both directions and preserve time order.
            node_neighbor_events[src].append((t, operation, dst_type, dst_msg))
            node_neighbor_events[dst].append((t, operation, src_type, src_msg))

        for node_id, events in node_neighbor_events.items():
            # Events are already in time order because we iterate over sorted_edges.
            prefixed_neighbors = []
            for _, edge_type, neighbor_type, neighbor_msg in events:
                display_type = "process" if neighbor_type == "subject" else neighbor_type
                clean_props = normalize_label_props(neighbor_msg)
                item = (
                    f"{edge_type} {display_type} {clean_props}"
                    if clean_props
                    else f"{edge_type} {display_type}"
                )
                prefixed_neighbors.append(item)
            # log(f"Node {node_id} neighbors (time-ordered): {prefixed_neighbors}")
            nodes[node_id].extend(prefixed_neighbors[:300])

        features, types, index_map = [], [], {}
        for node_id, props in nodes.items():
            features.append(props)
            types.append(node_types[node_id])
            index_map[node_id] = len(features) - 1

        data_of_graphs.append((features, types, list(index_map.keys())))

    token_cache = {}
    node2corpus = defaultdict(list)
    for graphs in log_tqdm(data_of_graphs, desc="Tokenizing corpus"):
        for msg, node_type, node_id in zip(graphs[0], graphs[1], graphs[2]):
            sentence = " ".join(msg)
            if sentence not in token_cache:
                # log(f'Sentence: {sentence}') # VERY BAD
                tokens = tokenize_flash_sentence(sentence)
                # log(f"Tokens: {tokens}")
                token_cache[sentence] = tokens

            tokens = token_cache[sentence]
            log(f"Node {node_id} ({node_type}) sentence: '{sentence}' -> tokens: {tokens}")
            break
            node2corpus[node_id].extend(tokens)

    return node2corpus


def _sample_dates_for_split(dates, sample_fraction, seed):
    if sample_fraction >= 1.0:
        return list(dates)

    if sample_fraction <= 0.0:
        return []

    if len(dates) == 0:
        return []

    sample_size = max(1, int(len(dates) * sample_fraction))
    sample_size = min(sample_size, len(dates))
    rng = random.Random(seed)
    return sorted(rng.sample(list(dates), sample_size))


def get_split_to_graph_paths(cfg, splits):
    argus_cfg = cfg.featurization.argus
    sample_fraction = float(getattr(argus_cfg, "sample_fraction", 1.0))
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(
            f"Invalid argus.sample_fraction={sample_fraction}. Expected value in (0, 1]."
        )

    base_seed = int(getattr(cfg.featurization, "seed", 0))
    split_to_paths = {}
    for split in splits:
        split_dates = list(getattr(cfg.dataset, f"{split}_dates"))
        split_seed = base_seed + sum(ord(ch) for ch in split)
        sampled_dates = _sample_dates_for_split(split_dates, sample_fraction, split_seed)
        if sample_fraction < 1.0:
            log(
                f"ARGUS sampling for split='{split}': "
                f"{len(sampled_dates)}/{len(split_dates)} dates "
                f"(fraction={sample_fraction:.3f})."
            )

        split_to_paths[split] = get_all_graphs_for_dates(
            cfg.transformation._graphs_dir,
            sampled_dates,
        )

    return split_to_paths


class RepeatableIterator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for phrase in self.data:
            yield phrase


class ArgusMLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


class StopOnLossCallback(TrainerCallback):
    def __init__(self, target_loss: float, min_steps_before_stop: int = 0):
        self.target_loss = target_loss
        self.min_steps_before_stop = min_steps_before_stop

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control

        if state.global_step < self.min_steps_before_stop:
            return control

        current_loss = float(logs["loss"])
        if current_loss <= self.target_loss:
            log(
                f"Stopping MLM adaptation early at step={state.global_step}, "
                f"epoch={state.epoch:.2f}, loss={current_loss:.4f} "
                f"(target={self.target_loss:.4f})."
            )
            control.should_training_stop = True
        return control


def main(cfg):
    log_start(__file__)

    argus_cfg = cfg.featurization.argus
    adapt_codebert = bool(getattr(argus_cfg, "adapt_codebert", True))
    epochs = cfg.featurization.epochs
    workers = argus_cfg.workers
    model_name = getattr(argus_cfg, "codebert_model_name", "microsoft/codebert-base")
    max_length = int(getattr(argus_cfg, "max_length", 512))
    mlm_probability = float(getattr(argus_cfg, "mlm_probability", 0.15))
    batch_size = int(getattr(argus_cfg, "mlm_batch_size", 8))
    learning_rate = float(getattr(argus_cfg, "mlm_learning_rate", 5e-5))
    target_loss = getattr(argus_cfg, "target_loss", None)
    min_steps_before_stop = int(getattr(argus_cfg, "min_steps_before_stop", 0))

    model_save_dir = cfg.featurization._model_dir
    os.makedirs(model_save_dir, exist_ok=True)

    log(
        f"ARGUS mode: adapt_codebert={adapt_codebert}, "
        f"model={model_name}, mlm_learning_rate={learning_rate}"
    )

    if not adapt_codebert:
        log(
            "Skipping CodeBERT domain adaptation (argus.adapt_codebert=False). "
            f"Feat inference will load base model: {model_name}."
        )
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning,
    )

    training_files = get_splits_to_train_featurization(cfg)
    node2corpus = get_node2corpus(cfg=cfg, splits=training_files)
    all_phrases = list(node2corpus.values())
    log(f"Total nodes to featurize: {len(all_phrases)}")

    sentences = [" ".join(tokens) for tokens in all_phrases if len(tokens) > 0]
    if len(sentences) == 0:
        raise ValueError("Argus corpus is empty. Cannot adapt CodeBERT without training sentences.")

    log(f"Loading CodeBERT tokenizer/model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    encodings = tokenizer(
        sentences,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True,
    )
    train_dataset = ArgusMLMDataset(encodings)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    log("Running CodeBERT domain adaptation with MLM...")
    mlm_tmp_dir = os.path.join(model_save_dir, "codebert_mlm_tmp")

    training_args = TrainingArguments(
        output_dir=mlm_tmp_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        dataloader_num_workers=workers,
        optim="adamw_torch",
        save_strategy="no",
        report_to=[],
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if target_loss is not None:
        target_loss = float(target_loss)
        if target_loss > 0:
            trainer.add_callback(StopOnLossCallback(target_loss, min_steps_before_stop))
            log(
                f"Enabled early stop for MLM adaptation at loss <= {target_loss:.4f} "
                f"(min_steps_before_stop={min_steps_before_stop})."
            )

    trainer.train()

    final_model_dir = os.path.join(model_save_dir, "codebert_mlm_final")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    log(f"Adapted CodeBERT model saved to {final_model_dir}")
