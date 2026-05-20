"""Data utilities for graph batching and temporal data handling.

Provides custom PyTorch Geometric data structures and utilities for:
- Temporal graph data collation and batching
- Graph reindexing for mini-batch training
- TGN memory and neighbor sampling integration
- Multi-dataset handling and preprocessing
"""

import copy
import gc
import math
import os
import pickle
from collections import defaultdict
from functools import cached_property

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, TemporalData
from torch_geometric.data.collate import collate
from torch_geometric.data.data import size_repr
from torch_geometric.data.temporal import prepare_idx
from torch_geometric.loader import TemporalDataLoader
from torch_scatter import scatter

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.debug_tests import debug_test_batching
from pidsmaker.encoders import TGNEncoder
from pidsmaker.tgn import LastNeighborLoader
from pidsmaker.utils.dataset_utils import (
    get_node_map,
    get_num_edge_type,
    get_rel2id,
    get_possible_events,
)
from pidsmaker.utils.utils import get_multi_datasets, log, log_dataset_stats, log_tqdm


# Float tensors on each batch worth fp16-ing for the lazy_batches disk format.
# Strictly the wide-d node/edge feature tensors — keep one-hot type vectors,
# timestamps, indices, and edge_feats (often msg-like) at native precision.
_LAZY_FP16_KEYS = (
    "msg",
    "x_src",
    "x_dst",
    "x",
    "x_from_tgn",
    "x_to_tgn",
    "x_tgn",
)


def _cast_lazy_keys(batch, dtype):
    """Cast each of the configured wide float tensors on ``batch`` to ``dtype``.

    No-op on keys that don't exist on the batch or are already in ``dtype``.
    Used to round-trip lazy-batch storage through a smaller on-disk dtype
    (float16) while keeping the runtime / model side at float32.
    """
    for k in _LAZY_FP16_KEYS:
        if k in batch._store:
            t = batch._store[k]
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype:
                batch._store[k] = t.to(dtype)


class LazyBatchList:
    """Disk-backed iterable replacing an in-RAM list of fully-prepared batches.

    Each batch is serialized to its own ``.pt`` file under ``dir`` and named
    ``batch_<idx:06d>.pt``. Iteration loads one batch at a time and yields
    it; nothing is cached between accesses, so peak resident memory is
    bounded by a single batch.

    Designed as a drop-in replacement for the inner ``data_list`` of
    ``train_data`` / ``val_data`` / ``test_data`` after all batching stages
    have run. The training/inference loops never need to know whether they
    are iterating an in-RAM list or one of these — same ``__iter__``,
    ``__len__``, ``__getitem__`` semantics.

    When ``full_data`` is provided, three TGN edge-feature tensors
    (``msg_tgn``, ``t_tgn``, ``edge_type_tgn``) are *not* saved per batch —
    only the ``e_id_tgn`` index — and they get hydrated from ``full_data``
    at iteration time. This is the dominant on-disk saving for big TGN
    runs (msg_tgn alone is typically ~10MB/batch).
    """

    def __init__(self, directory, num_batches, full_data=None, fp16=False):
        self._dir = directory
        self._n = int(num_batches)
        # Holding full_data here keeps the mmap-backed tensors alive for the
        # lifetime of this list, so `del full_data` upstream is harmless.
        self._full_data = full_data
        # When True, the wide float tensors on disk are float16 and must be
        # upcast to float32 after every load before yielding the batch.
        self._fp16 = bool(fp16)

    def __len__(self):
        return self._n

    def _path(self, idx):
        return os.path.join(self._dir, f"batch_{idx:06d}.pt")

    def _hydrate(self, batch):
        """Re-attach TGN edge features from full_data using batch.e_id_tgn,
        and upcast fp16-on-disk wide floats back to float32.

        No-op for fields not present or already at the expected dtype.
        """
        if self._fp16:
            _cast_lazy_keys(batch, torch.float32)
        if self._full_data is None or not hasattr(batch, "e_id_tgn"):
            return batch
        e_id = batch.e_id_tgn
        # `.clone().contiguous()` detaches each slice from the mmap so the
        # GPU transfer downstream doesn't have to fault file pages on every
        # forward pass; the rehydrated tensor is in normal pageable RAM.
        if not hasattr(batch, "msg_tgn"):
            batch.msg_tgn = torch.as_tensor(self._full_data.msg[e_id]).clone().contiguous()
        if not hasattr(batch, "t_tgn"):
            batch.t_tgn = torch.as_tensor(self._full_data.t[e_id]).clone().contiguous()
        if not hasattr(batch, "edge_type_tgn"):
            batch.edge_type_tgn = (
                torch.as_tensor(self._full_data.edge_type[e_id]).clone().contiguous()
            )
        return batch

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._n))]
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        return self._hydrate(torch.load(self._path(idx), map_location="cpu"))

    def __iter__(self):
        for i in range(self._n):
            yield self._hydrate(torch.load(self._path(i), map_location="cpu"))

    def __bool__(self):
        return self._n > 0

    @staticmethod
    def from_list(batches, directory, fp16=False):
        """Serialize each batch to its own file under ``directory`` and return
        a proxy. The caller is responsible for releasing the original list
        references after this returns.
        """
        os.makedirs(directory, exist_ok=True)
        for i, b in enumerate(batches):
            if fp16:
                _cast_lazy_keys(b, torch.float16)
            torch.save(b, os.path.join(directory, f"batch_{i:06d}.pt"))
        return LazyBatchList(directory, len(batches), fp16=fp16)


def _wrap_split_lazy(split, base_dir, split_name, fp16=False):
    """Replace each inner batch list in a split with a LazyBatchList.

    A split is ``list[list[batch]]`` — outer is the dataset group index
    (always length 1 for single-dataset; longer for multi-dataset).
    Returns the new outer list and frees the input as it goes so the
    caller's RAM doesn't carry both copies.
    """
    out = []
    for group_idx, data_list in enumerate(split):
        group_dir = os.path.join(base_dir, split_name, f"group_{group_idx}")
        out.append(LazyBatchList.from_list(data_list, group_dir, fp16=fp16))
        # drop in-place so caller's `split` variable releases its refs as we go
        split[group_idx] = None
    return out


class LazyTestData:
    """Disk-backed proxy for a fully-processed test_data structure.

    `test_data` is a `list[list[CollatableTemporalData]]` (outer list = per
    dataset group, inner list = batches). On huge datasets keeping it in
    RAM through the entire training loop dwarfs the model itself. This
    proxy serialises it once to disk, drops the in-RAM copy, and re-loads
    it on iteration. Call `release()` after evaluation to free the cache.

    Iteration semantics are identical to the underlying list, so callers
    (inference_loop) need no changes.
    """

    def __init__(self, path):
        self._path = path
        self._cached = None

    def _load(self):
        if self._cached is None:
            self._cached = torch.load(self._path, map_location="cpu")
        return self._cached

    def release(self):
        """Drop the cached payload so subsequent iteration re-reads from disk."""
        self._cached = None
        gc.collect()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __getitem__(self, i):
        return self._load()[i]

    def __bool__(self):
        # Keep `if test_data:` checks working without forcing a load.
        return True


class CollatableTemporalData(TemporalData):
    """
    We use this class instead of TemporalData in order to easily concatenate data
    objects together without any batching behavior.
    Normal TemporalData doesn't support edge_index so we define it here.
    """

    def __init__(
        self,
        src=None,
        dst=None,
        t=None,
        msg=None,
        **kwargs,
    ):
        super().__init__(src=src, dst=dst, t=t, msg=msg, **kwargs)
        self.tgn_mode = False

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "original_edge_index":  # used to retrieve original node IDs during evaluation
            return 0
        if "edge_index" in key or key in ["src", "dst", "reindexed_original_n_id_tgn"]:
            if self.tgn_mode:
                return torch.unique(self.n_id_tgn).numel()
            return self.num_nodes
        return 0

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        return 1 if "edge_index" in key else 0

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ", ".join([size_repr(k, v) for k, v in self._store.items()])
        info += ", " + size_repr("edge_index", self.edge_index)
        return f"{cls}({info})"

    def index_select(self, idx):
        """ "Indexing to handle (2, E) index attributes"""
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.size(0) == self.num_events:
                data[key] = value[idx]
            elif value.ndim == 2 and value.size(1) == self.num_events and "index" in key:
                data[key] = value[:, idx]
        return data

    @property
    def num_nodes(self):
        if "original_n_id" in self:
            return self.original_n_id.numel()
        return self.edge_index.unique().numel()

    @cached_property
    def node_type_argmax(self):
        return self.node_type.max(dim=1).indices

    @cached_property
    def node_type_src_argmax(self):
        return self.node_type_src.max(dim=1).indices

    @cached_property
    def edge_type_argmax(self):
        return self.edge_type.max(dim=1).indices

    @cached_property
    def node_type_dst_argmax(self):
        return self.node_type_dst.max(dim=1).indices


def load_all_datasets(cfg, device, only_keep=None):
    multi_dataset = cfg.batching.multi_dataset_training
    train_data = load_data_set(cfg, split="train", multi_dataset=multi_dataset)
    val_data = load_data_set(cfg, split="val", multi_dataset=multi_dataset)
    test_data = load_data_set(cfg, split="test", multi_dataset=False)

    if only_keep is not None:
        train_data = train_data[:only_keep]
        val_data = val_data[:only_keep]
        test_data = test_data[:only_keep]

    # TGN's last-neighbor batching is the only consumer of the heavy fields
    # (msg / t / edge_type) on `full_data`. For non-TGN paths we don't need
    # full_data at all — max_node is streamed from per-graph tensors below.
    intra_methods = cfg.batching.intra_graph_batching.used_methods or ""
    use_tgn_last_neighbor = "tgn_last_neighbor" in intra_methods

    # Compute max_node without building a full concatenated copy of src/dst.
    max_node = compute_max_node([train_data, val_data, test_data])
    print(f"Max node in {cfg.dataset.name}: {max_node}")

    if use_tgn_last_neighbor:
        full_keys = ["msg", "t", "edge_type", "src", "dst"]
        if getattr(cfg.batching, "mmap_full_data", None):
            mmap_dir = os.path.join(cfg.batching._preprocessed_graphs_dir, "full_data_mmap")
            log(f"[mmap_full_data] Building disk-backed full_data at {mmap_dir}.")
            full_data = build_full_data_mmap(
                [train_data, val_data, test_data], keys=full_keys, out_dir=mmap_dir
            )
        else:
            full_data = get_full_data([train_data, val_data, test_data], keys=full_keys)
    else:
        full_data = None

    # NOTE: a previous version of this function tried to del `g.msg` whenever
    # TGN memory was not in use. That is unsafe — `model.embed()` always
    # passes `msg=batch.msg` to the encoder regardless of whether the encoder
    # consumes it (the kwarg access itself raises AttributeError when the
    # attribute is missing). Keep per-graph msg around; the real savings
    # come from skipping it in `full_data` (above) and from `mmap_full_data`.

    graph_reindexer = GraphReindexer(
        device=device,
        num_nodes=max_node,
        fix_buggy_graph_reindexer=cfg.batching.fix_buggy_graph_reindexer,
    )

    # Global batching (unique edge type batches, fixed-size edge length)
    datasets = run_global_batching(train_data, val_data, test_data, cfg, device)
    del train_data, val_data, test_data
    gc.collect()

    # Intra graph batching (TGN 1024 batches, last neighbor loader)
    datasets = run_intra_graph_batching(datasets, full_data, device, max_node, cfg, graph_reindexer)
    del full_data
    gc.collect()

    # Reindexing stuff (create node-level attributes)
    datasets = run_reindexing_preprocessing(datasets, graph_reindexer, device, cfg)

    # Inter graph batching (actual mini-batching of very small graphs)
    datasets = run_inter_graph_batching(datasets, cfg)

    train_data, val_data, test_data = datasets

    # Optionally swap every prepared batch out to disk and replace the
    # in-RAM lists with LazyBatchList proxies. After all batching stages
    # the precomputed TGN neighbor tensors per batch (msg_tgn, t_tgn,
    # x_from_tgn, x_to_tgn, x_tgn, node_type_tgn, edge_type_tgn, ...) live
    # in CPU RAM for the entire training run — on TGN systems this is
    # often 100-200GB. Lazy batches cut that to a single batch's worth.
    #
    # For the TGN path the streaming has already happened inside
    # compute_tgn_graphs (so batches were never in RAM all at once); the
    # already-LazyBatchList groups are detected and skipped here.
    if getattr(cfg.batching, "lazy_batches", None):
        base_dir = os.path.join(cfg.batching._preprocessed_graphs_dir, "lazy_batches")
        lazy_fp16 = bool(getattr(cfg.batching, "lazy_batches_fp16", False))

        def _already_lazy(split):
            return all(isinstance(g, LazyBatchList) for g in split)

        if not _already_lazy(train_data):
            log(f"[lazy_batches] Serialising prepared train batches to {base_dir}.")
            train_data = _wrap_split_lazy(train_data, base_dir, "train", fp16=lazy_fp16)
            gc.collect()
        if not _already_lazy(val_data):
            log(f"[lazy_batches] Serialising prepared val batches to {base_dir}.")
            val_data = _wrap_split_lazy(val_data, base_dir, "val", fp16=lazy_fp16)
            gc.collect()
        if not _already_lazy(test_data):
            log(f"[lazy_batches] Serialising prepared test batches to {base_dir}.")
            test_data = _wrap_split_lazy(test_data, base_dir, "test", fp16=lazy_fp16)
            gc.collect()
        total_batches = sum(len(g) for split in (train_data, val_data, test_data) for g in split)
        log(f"[lazy_batches] Active; {total_batches} batch(es) on disk (fp16={lazy_fp16}).")

    # Optionally serialise test_data to disk so it can be dropped from RAM
    # during the long training loop. Re-loaded on-demand by inference_loop
    # via LazyTestData iteration semantics.
    # NOTE: with lazy_batches on, test_data is already disk-backed per batch,
    # so this extra step would just create a redundant full-split pickle.
    if getattr(cfg.batching, "lazy_test_data", None) and not getattr(
        cfg.batching, "lazy_batches", None
    ):
        out_dir = cfg.batching._preprocessed_graphs_dir
        os.makedirs(out_dir, exist_ok=True)
        test_path = os.path.join(out_dir, "test_data.lazy.pkl")
        log(f"[lazy_test_data] Saving test_data to {test_path} and freeing it from RAM.")
        torch.save(test_data, test_path)
        del test_data
        gc.collect()
        test_data = LazyTestData(test_path)

    return train_data, val_data, test_data, max_node


def load_data_list(path, split, cfg):
    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        data = torch.load(filepath).to("cpu")
        data_list.append(data)

    data_list = extract_msg_from_data(data_list, cfg)
    return data_list


def load_data_set(cfg, split: str, multi_dataset=False) -> list[CollatableTemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """
    if multi_dataset:
        multi_datasets = get_multi_datasets(cfg)
        all_data_lists = []
        for dataset in multi_datasets:
            updated_cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)
            path = updated_cfg.feat_inference._edge_embeds_dir
            all_data_lists.append(load_data_list(path, split, cfg))
        return all_data_lists

    else:
        path = cfg.feat_inference._edge_embeds_dir
        return [load_data_list(path, split, cfg)]


def extract_msg_from_data(
    data_set: list[CollatableTemporalData], cfg
) -> list[CollatableTemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.featurization.emb_dim
    only_type = cfg.featurization.used_method.strip() == "only_type"
    only_ones = cfg.featurization.used_method.strip() == "only_ones"
    if only_type or only_ones or emb_dim is None:
        emb_dim = 0
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    selected_node_feats = cfg.batching.node_features

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim * 2) + (node_type_dim * 2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(
            f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len} (Is your num_edge_types correct?)"
        )

    field_to_size = {
        "src_type": node_type_dim,
        "src_emb": emb_dim,
        "edge_type": edge_type_dim,
        "dst_type": node_type_dim,
        "dst_emb": emb_dim,
    }

    if "edges_distribution" in selected_node_feats:
        max_num_nodes = max([torch.cat([g.src, g.dst]).max().item() for g in data_set]) + 1
        x_distrib = torch.zeros(max_num_nodes, edge_type_dim * 2, dtype=torch.float)

    if only_type:
        selected_node_feats = ["node_type"]
    elif only_ones:
        selected_node_feats = ["only_ones"]
    else:
        selected_node_feats = list(
            map(lambda x: x.strip(), selected_node_feats.replace("-", ",").split(","))
        )

    edge_features = list(map(lambda x: x.strip(), cfg.batching.edge_features.split(",")))
    possible_triplets = get_possible_triplets(cfg) if "edge_type_triplet" in edge_features else None

    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size.items():
            fields[field] = g.msg[:, idx : idx + size]
            idx += size

        # Selects only the node features we want
        x_src, x_dst = [], []
        for feat in selected_node_feats:
            if feat == "node_emb":
                x_src.append(fields["src_emb"])
                x_dst.append(fields["dst_emb"])

            elif feat == "node_type":
                x_src.append(fields["src_type"])
                x_dst.append(fields["dst_type"])

            elif feat == "edges_distribution":  # as in ThreaTrace
                x_distrib.scatter_add_(
                    0, g.src.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"]
                )
                x_distrib[:, edge_type_dim:].scatter_add_(
                    0, g.dst.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"]
                )

                # In ThreaTrace they don't standardize, here we do standardize by max value in TW
                x_distrib = x_distrib / (x_distrib.max() + 1e-12)

                x_src.append(x_distrib[g.src])
                x_dst.append(x_distrib[g.dst])

                x_distrib.fill_(0)

            elif feat == "only_ones":
                x_src.append(fields["src_type"].clone().fill_(1))
                x_dst.append(fields["dst_type"].clone().fill_(1))

            else:
                raise ValueError(f"Node feature {feat} is invalid.")

        x_src = torch.cat(x_src, dim=-1)
        x_dst = torch.cat(x_dst, dim=-1)

        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)

        num_edge_types = get_num_edge_type(cfg)
        edge_feats = build_edge_feats(fields, msg, edge_features, possible_triplets, num_edge_types)

        edge_type = (
            get_triplet_edge_types(
                fields["src_type"],
                fields["dst_type"],
                fields["edge_type"],
                possible_triplets,
                num_edge_types,
            )
            if "edge_type_triplet" in edge_features
            else fields["edge_type"]
        )

        g.x_src = x_src
        g.x_dst = x_dst
        g.edge_feats = edge_feats
        # node_type_src/dst (and edge_type when not using triplets) are views into the
        # original g.msg. Cloning detaches them so the wide msg can be freed later by
        # callers (e.g. once `full_data` is built and per-graph msg is no longer needed).
        g.edge_type = edge_type if edge_type is not fields["edge_type"] else edge_type.clone()
        g.node_type_src = fields["src_type"].clone()
        g.node_type_dst = fields["dst_type"].clone()

        if "tgn" in cfg.training.encoder.used_methods and cfg.training.encoder.tgn.use_memory:
            g.msg = msg  # smaller rebuilt msg; original wide msg can now be GC'd

        # NOTE: do not add edge_index as it is already within `CollatableTemporalData`
        # g.edge_index = ...

    return data_set

def get_possible_triplets(cfg):
    entity_map = get_node_map(from_zero=True)
    event_map = get_rel2id(cfg, from_zero=True)
    possible_events = get_possible_events(cfg)

    possible_triplets = [
        [entity_map[src_type], entity_map[dst_type], event_map[event]]
        for (src_type, dst_type), events in possible_events.items()
        for event in events
    ]
    return torch.tensor(possible_triplets, dtype=torch.long)


def get_triplet_edge_types(src_type, dst_type, edge_type, possible_triplets, num_edge_types):
    triplets = torch.stack(
        (src_type.max(dim=1).indices, dst_type.max(dim=1).indices, edge_type.max(dim=1).indices),
        dim=1,
    )
    matches = (triplets.unsqueeze(1) == possible_triplets.unsqueeze(0)).all(dim=2)
    return F.one_hot(matches.long().argmax(dim=1), num_classes=num_edge_types).to(torch.float)


def build_edge_feats(fields, msg, edge_features, possible_triplets, num_edge_types):
    edge_feats = []
    if "edge_type" in edge_features:
        edge_feats.append(fields["edge_type"])
    if "edge_type_triplet" in edge_features:
        triplets = get_triplet_edge_types(
            fields["src_type"],
            fields["dst_type"],
            fields["edge_type"],
            possible_triplets,
            num_edge_types,
        )
        edge_feats.append(triplets)
    if "msg" in edge_features:
        edge_feats.append(msg)
    edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
    return edge_feats


def compute_max_node(datasets):
    """Stream max(src, dst) across every per-graph tensor.

    Replaces ``torch.cat([full_data.src, full_data.dst]).max()`` which built
    a full extra copy of every src/dst tensor just to find the max. Returns
    ``max + 1`` so it matches the previous +1 convention.
    """
    max_node = -1
    for dataset_group in datasets:
        for dataset in dataset_group:
            for data in dataset:
                if data.src.numel():
                    m = int(data.src.max().item())
                    if m > max_node:
                        max_node = m
                if data.dst.numel():
                    m = int(data.dst.max().item())
                    if m > max_node:
                        max_node = m
    return max_node + 1


def build_full_data_mmap(datasets, keys, out_dir):
    """Concatenate per-graph attributes into disk-backed memory-mapped tensors.

    Each key gets its own raw binary file written sequentially, one per-graph
    slice at a time. Peak RAM is bounded by a single per-graph tensor — the
    full concatenated copy never exists in memory. The returned ``Data``
    exposes ``np.memmap``-backed torch tensors; indexing (e.g.
    ``full_data.msg[e_id]``) pages in the requested rows on demand and the
    gathered slice gets a new in-RAM allocation only for the gathered rows.

    Uses numpy.memmap rather than torch.UntypedStorage.from_file so we stay
    compatible with the torch 1.13 pin in this project's Dockerfile.
    """
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # First pass: collect dtype, tail shape, total length for every key.
    sample_per_key = {}
    total_E = 0
    for dataset_group in datasets:
        for dataset in dataset_group:
            for data in dataset:
                total_E += int(data.src.numel())
                for k in keys:
                    if k not in sample_per_key:
                        sample_per_key[k] = getattr(data, k)

    full_data = Data()
    full_data._mmap_arrays = []  # keep numpy memmap refs alive

    for k in keys:
        sample = sample_per_key[k]
        torch_dtype = sample.dtype
        tail_shape = tuple(int(s) for s in sample.shape[1:])
        path = os.path.join(out_dir, f"full_{k}.bin")

        # Second pass: stream per-graph slices to disk. Each iteration touches
        # one tensor at a time so peak extra RAM is bounded by the largest
        # single per-graph tensor (not the full concatenation).
        with open(path, "wb") as f:
            for dataset_group in datasets:
                for dataset in dataset_group:
                    for data in dataset:
                        t = getattr(data, k).contiguous()
                        # `.numpy()` is zero-copy; tobytes copies once, then is
                        # GC'd as the next iteration starts.
                        f.write(t.numpy().tobytes())

        # Read back as a memory-mapped numpy array, then wrap as a torch
        # tensor. torch.from_numpy shares memory with the memmap, so element
        # access pages in from disk; no full-in-RAM copy is made.
        shape = (total_E,) + tail_shape if tail_shape else (total_E,)
        np_dtype = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }[torch_dtype]
        arr = np.memmap(path, dtype=np_dtype, mode="r", shape=shape)
        full_data._mmap_arrays.append(arr)
        setattr(full_data, k, torch.from_numpy(arr))

    log(f"[mmap_full_data] Wrote {len(keys)} key(s) totalling {total_E} edges to {out_dir}.")
    return full_data


def get_full_data(datasets, keys=None):
    """Concatenate per-graph attributes into a single Data object.

    Only the keys actually consumed downstream are gathered (default: msg, t,
    edge_type, src, dst — `node_type_src/dst` were collected historically but
    never read). The narrower set lets callers skip work entirely when they
    don't need the TGN-only fields.
    """
    if keys is None:
        keys = ["msg", "t", "edge_type", "src", "dst"]
    all_data = {k: [] for k in keys}
    for dataset_group in datasets:
        for dataset in dataset_group:
            for data in dataset:
                for k in all_data:
                    all_data[k].append(getattr(data, k))

    full_data = Data(**{k: torch.cat(v) for k, v in all_data.items()})
    # Free the per-key Python lists holding view references onto every graph
    # before returning (frees one extra reference each, allowing earlier GC).
    all_data.clear()
    return full_data


def custom_temporal_data_loader(data: CollatableTemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        yield batch


def temporal_data_to_data(data: CollatableTemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `CollatableTemporalData` to `Data` before using it.
    """
    data = Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})
    del data.num_nodes
    return data


def collate_temporal_data(data_list: list[CollatableTemporalData]) -> CollatableTemporalData:
    """
    Concatenates attributes from data ojects into a single data object.
    Do not use with `Data` directly because it will use batching when collating.
    """
    assert all([not isinstance(data, Data) for data in data_list]), (
        "Concatenating Data objects result in batching."
    )

    data = collate(CollatableTemporalData, data_list, increment=False)[0]
    del data.ptr
    del data.batch

    return data


def batch_temporal_data(
    data: CollatableTemporalData, batch_size: int, batch_mode: str, cfg, device
) -> list[CollatableTemporalData]:
    if batch_mode == "edges":
        num_batches = math.ceil(
            len(data.src) / batch_size
        )  # NOTE: the last batch won't have the same number of edges as the batch

        data_list = [
            data[int(i * batch_size) : int((i + 1) * batch_size)] for i in range(num_batches)
        ]
        return data_list

    elif batch_mode == "minutes":
        window_length_ns = int(cfg.construction.time_window_size * 60_000_000_000)
        sliding_ns = int(batch_size * 60_000_000_000)  # min to ns

        t = data.t
        t0 = t.min()
        t0_aligned = (t0 // sliding_ns) * sliding_ns

        # Compute window indices for all data points
        relative_t = t - t0_aligned
        window_indices = relative_t // sliding_ns

        # Since data.t is sorted, find boundaries of unique window indices
        unique_windows, counts = torch.unique(window_indices, return_counts=True)
        cum_counts = torch.cumsum(counts, dim=0)
        start_indices = torch.cat([torch.tensor([0], device=cum_counts.device), cum_counts[:-1]])
        end_indices = cum_counts

        # Create windows by slicing data
        windows = []
        for start, end, window_idx in zip(start_indices, end_indices, unique_windows):
            if end <= start:  # Skip empty windows
                continue

            # Get indices for the current window
            indices = torch.arange(start, end, device=t.device)

            # Filter points within exact window time range
            window_start = t0_aligned + window_idx * sliding_ns
            window_end = window_start + window_length_ns
            mask = (t[indices] >= window_start) & (t[indices] < window_end)
            window_indices_final = indices[mask]

            if len(window_indices_final) == 0:
                continue

            # Slice the original data using the filtered indices
            window_data = data[window_indices_final]
            windows.append(window_data)

        return windows

    elif batch_mode == "unique_edge_types":
        partitions = []
        seen_edges = defaultdict(set)
        data.to(device)

        partitions = []
        src_list = data.src.tolist()
        dst_list = data.dst.tolist()
        type_list = data.edge_type.max(dim=1).indices.tolist()
        start, end = 0, 0

        for i, (src, dst, edge_type) in log_tqdm(
            enumerate(zip(src_list, dst_list, type_list)),
            desc="Generating unique edge type batches",
        ):
            if (src, dst) in seen_edges:
                # Conflict: (src, dst) already exists with a different edge_type
                partitions.append(data[start:end])
                start = end
                end += 1
                seen_edges = defaultdict(set)
            else:
                end += 1

            seen_edges[(src, dst)].add(edge_type)

        # Add the last partition if not empty
        if end > start:
            partitions.append(data[start:end])

        data.to("cpu")
        return partitions

    raise ValueError(f"Invalid or missing batch mode {batch_mode}")


def run_global_batching(train_data, val_data, test_data, cfg, device):
    # Concatenates all data into a single data so that iterating over batches
    # of edges is more consistent with TGN
    global_batching_cfg = cfg.batching.global_batching
    batch_mode = global_batching_cfg.used_method
    bs = global_batching_cfg.global_batching_batch_size
    bs_inference = global_batching_cfg.global_batching_batch_size_inference
    if batch_mode != "none":
        if (bs not in [None, 0]) or batch_mode == "unique_edge_types":
            train_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in train_data
            ]
            val_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in val_data
            ]
            test_data = [
                batch_temporal_data(collate_temporal_data(graphs), bs, batch_mode, cfg, device)
                for graphs in test_data
            ]

        elif bs_inference not in [None, 0]:
            test_data = [
                batch_temporal_data(
                    collate_temporal_data(graphs), bs_inference, batch_mode, cfg, device
                )
                for graphs in test_data
            ]

    return train_data, val_data, test_data


def run_reindexing_preprocessing(datasets, graph_reindexer, device, cfg):
    use_unique_edge_types = "unique_edge_types" in cfg.batching.global_batching.used_method
    if not use_unique_edge_types:
        log_dataset_stats(datasets)
        # By default we only have x_src and x_dst of shape (E, d), here we create x of shape (N, d)
        use_tgn = "tgn" in cfg.training.encoder.used_methods
        reindex_graphs(
            datasets,
            graph_reindexer,
            device,
            use_tgn,
            x_is_tuple=cfg.training.encoder.x_is_tuple,
        )

    return datasets


def run_intra_graph_batching(datasets, full_data, device, max_node, cfg, graph_reindexer):
    def standard_intra_batching(dataset, method):
        result = []
        for data_list in dataset:
            result.append([])
            for batch in log_tqdm(data_list, desc="Creating TGN batches"):
                # Use temporal batch loader used in TGN
                if method == "edges":
                    batch_size = cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
                    batch_loader = custom_temporal_data_loader(batch, batch_size=batch_size)
                elif method == "neighbor_sampling":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Invalid sampling method {method}")

                for small_batch in batch_loader:
                    result[-1].append(small_batch)
        return result

    methods = map(
        lambda x: x.strip(),
        cfg.batching.intra_graph_batching.used_methods.split(","),
    )
    for method in methods:
        if method == "none":
            continue

        elif method in ["edges", "neighbor_sampling"]:
            datasets = [standard_intra_batching(dataset, method) for dataset in datasets]

        elif method == "tgn_last_neighbor":
            tgn_loader_cfg = cfg.batching.intra_graph_batching.tgn_last_neighbor
            sample = datasets[0][0][0]
            # If lazy_batches is requested, stream-save each fattened batch to
            # disk inside compute_tgn_graphs so peak resident RAM stays bounded
            # by a single batch instead of the whole dataset.
            lazy_dir = None
            if getattr(cfg.batching, "lazy_batches", None):
                lazy_dir = os.path.join(
                    cfg.batching._preprocessed_graphs_dir, "lazy_batches"
                )
                os.makedirs(lazy_dir, exist_ok=True)
            lazy_fp16 = bool(getattr(cfg.batching, "lazy_batches_fp16", False))
            datasets = compute_tgn_graphs(
                datasets=datasets,
                full_data=full_data,
                graph_reindexer=graph_reindexer,
                device=device,
                max_node=max_node,
                tgn_loader_cfg=tgn_loader_cfg,
                node_feat_dim=sample.x_src.shape[1],
                node_type_dim=sample.node_type_src.shape[1],
                lazy_dir=lazy_dir,
                split_names=["train", "val", "test"],
                lazy_fp16=lazy_fp16,
            )

        else:
            raise ValueError(f"Invalid sampling method {method}")

    return datasets


def compute_tgn_graphs(
    datasets,
    full_data,
    graph_reindexer,
    device,
    max_node,
    tgn_loader_cfg,
    node_feat_dim,
    node_type_dim,
    lazy_dir=None,
    split_names=None,
    lazy_fp16=False,
):
    """Compute per-batch TGN neighbor tensors.

    When ``lazy_dir`` is provided, each batch is serialised to disk
    immediately after its *_tgn fields are set and the in-memory reference
    is dropped. The caller receives a structure of LazyBatchList proxies
    instead of fattened lists, so peak CPU RAM during this pass is bounded
    by a single batch — not the entire dataset (~20MB * 50K batches → 1TB
    on optc_h501-class data was previously the crash mode here).

    The optional ``split_names`` is a per-dataset-group list of names used
    to subdirectory the saved batches; if omitted, generic indices are
    used. Returned shape mirrors the input: ``list[list-or-LazyBatchList]``.
    """
    tgn_neighbor_n_hop = tgn_loader_cfg.tgn_neighbor_n_hop
    fix_tgn_neighbor_loader = tgn_loader_cfg.fix_tgn_neighbor_loader
    fix_buggy_orthrus_TGN = tgn_loader_cfg.fix_buggy_orthrus_TGN
    insert_neighbors_before = tgn_loader_cfg.insert_neighbors_before
    neighbor_size = tgn_loader_cfg.tgn_neighbor_size
    directed = tgn_loader_cfg.directed

    neighbor_loader = LastNeighborLoader(
        max_node, size=neighbor_size, directed=directed, device=device
    )

    node_feat_cache = torch.zeros((max_node, node_feat_dim), device=device)
    node_type_cache = torch.zeros((max_node, node_type_dim), device=device)
    assoc = torch.empty(max_node, dtype=torch.long, device=device)

    new_datasets = []
    for ds_idx, dataset in enumerate(datasets):
        split_name = (split_names[ds_idx] if split_names else f"ds_{ds_idx}") if lazy_dir else None
        new_groups = []
        for grp_idx, data_list in enumerate(dataset):
            group_dir = (
                os.path.join(lazy_dir, split_name, f"group_{grp_idx}") if lazy_dir else None
            )
            if group_dir is not None:
                os.makedirs(group_dir, exist_ok=True)
            saved_count = 0

            iter_idx = 0
            for batch in log_tqdm(data_list, desc="Computing TGN last neighbor graphs"):
                batch = batch.to(device)
                batch_edge_index = batch.edge_index.clone()
                src, dst = batch_edge_index

                if insert_neighbors_before:
                    neighbor_loader.insert(src, dst)

                n_id = batch_edge_index.unique()
                for _ in range(tgn_neighbor_n_hop):
                    n_id, edge_index, e_id = neighbor_loader(n_id)

                if fix_tgn_neighbor_loader:
                    # NOTE: TGN's loader wrongly index edges (less than 1% in the returned e_id and edge_index)
                    # https://github.com/pyg-team/pytorch_geometric/issues/10100
                    # Should be replaced by an actual fix when available
                    real_src = full_data.src[e_id.cpu()].to(device)
                    real_dst = full_data.dst[e_id.cpu()].to(device)

                    loader_src = n_id[edge_index[0]]
                    loader_dst = n_id[edge_index[1]]

                    match_dir1 = real_src.eq(loader_src) & real_dst.eq(loader_dst)
                    match_dir2 = real_src.eq(loader_dst) & real_dst.eq(loader_src)

                    valid_edges = match_dir1 | match_dir2
                    edge_index = edge_index[:, valid_edges]
                    e_id = e_id[valid_edges]

                num_nodes = n_id.size(
                    0
                )  # Important, this one is used as __inc__ when batching graphs
                assoc[n_id] = torch.arange(num_nodes, device=device)
                node_feat_cache[torch.cat([src, dst])] = torch.cat([batch.x_src, batch.x_dst])
                node_type_cache[torch.cat([src, dst])] = torch.cat(
                    [batch.node_type_src, batch.node_type_dst]
                )

                if fix_buggy_orthrus_TGN:
                    x_src = torch.zeros((num_nodes, node_feat_dim), device=device)
                    x_dst = x_src.clone()
                    src_id, dst_id = edge_index[0].unique(), edge_index[1].unique()
                    x_src[src_id] = node_feat_cache[n_id[src_id]]
                    x_dst[dst_id] = node_feat_cache[n_id[dst_id]]
                    new_x = node_feat_cache[n_id]
                    batch.x_from_tgn = x_src  # (N, d)
                    batch.x_to_tgn = x_dst  # (N, d)
                    batch.x_tgn = new_x  # (N, d)

                else:
                    (x_src, x_dst), *_ = graph_reindexer._reindex_graph(
                        batch_edge_index,
                        batch.x_src,
                        batch.x_dst,
                        max_num_node=num_nodes,
                        x_is_tuple=True,
                    )
                    batch.x_from_tgn = x_src
                    batch.x_to_tgn = x_dst
                    batch.x_tgn = x_src

                batch.tgn_mode = True
                batch.original_edge_index = batch_edge_index
                batch.original_n_id = batch_edge_index.unique()
                batch.reindexed_original_n_id_tgn = assoc[batch.original_n_id]
                batch.n_id_tgn = n_id
                batch.edge_index_tgn = edge_index
                batch.reindexed_edge_index_tgn = assoc[batch.edge_index]
                batch.node_type_tgn = node_type_cache[n_id]
                if group_dir is not None:
                    # Stream-save mode: keep just the global edge indices on
                    # the batch so LazyBatchList can rehydrate msg_tgn /
                    # t_tgn / edge_type_tgn from full_data at iteration time.
                    # msg_tgn alone is ~10MB/batch — skipping it is the
                    # difference between lazy_batches needing TBs of disk
                    # and fitting in a normal scratch budget.
                    batch.e_id_tgn = e_id.cpu()
                else:
                    batch.msg_tgn = full_data.msg[e_id.cpu()]
                    batch.t_tgn = full_data.t[e_id.cpu()]
                    batch.edge_type_tgn = full_data.edge_type[e_id.cpu()]

                batch = batch.to("cpu")

                if not insert_neighbors_before:
                    neighbor_loader.insert(src, dst)

                if group_dir is not None:
                    # Stream-save mode: persist this batch then null out the
                    # original list slot so the in-RAM reference can be
                    # reclaimed. Without this the batch lives in `data_list`
                    # for the rest of the loop and the per-batch *_tgn
                    # payload accumulates — which is exactly the OOM we are
                    # trying to avoid.
                    if lazy_fp16:
                        _cast_lazy_keys(batch, torch.float16)
                    torch.save(batch, os.path.join(group_dir, f"batch_{saved_count:06d}.pt"))
                    saved_count += 1
                    data_list[iter_idx] = None
                    batch = None
                iter_idx += 1

            if group_dir is not None:
                # In-place clear the now-mostly-None list and replace with the
                # disk-backed proxy at the upstream slot. We hand the proxy a
                # reference to full_data so it can re-hydrate the skipped TGN
                # edge features on iteration (and so `del full_data` upstream
                # doesn't reclaim it while we still need it).
                data_list.clear()
                new_groups.append(
                    LazyBatchList(
                        group_dir, saved_count, full_data=full_data, fp16=lazy_fp16
                    )
                )
                gc.collect()
            else:
                new_groups.append(data_list)
        new_datasets.append(new_groups)

    return new_datasets


def run_inter_graph_batching(datasets, cfg):
    def inter_batching(dataset, method):
        if method == "none":
            return dataset

        elif method == "graph_batching":
            bs = cfg.batching.inter_graph_batching.inter_graph_batch_size
            result = []
            for data_list in dataset:
                result.append([])
                for i in log_tqdm(
                    range(0, len(data_list), bs),
                    total=math.ceil(len(data_list) / bs),
                    desc="Mini-batching",
                ):
                    batch = data_list[i : i + bs]
                    data = collate(CollatableTemporalData, data_list=batch)[0]

                    use_tgn = "tgn" in cfg.training.encoder.used_methods
                    if cfg._debug and use_tgn:
                        debug_test_batching(batch, data, cfg)
                    result[-1].append(data)
            return result

        raise ValueError(f"Invalid inter-graph batching method {method}")

    method = cfg.batching.inter_graph_batching.used_method
    datasets = [inter_batching(dataset, method) for dataset in datasets]
    return datasets


class _Cache:
    def __init__(self, shape, device):
        self._cache = torch.zeros(shape, device=device)

    @property
    def cache(self):
        return self._cache

    def detach(self):
        self._cache = self._cache.detach()

    def to(self, device):
        self._cache = self._cache.to(device)
        return self


class GraphReindexer:
    """
    Simply transforms an edge_index and its src/dst node features of shape (E, d)
    to a reindexed edge_index with node IDs starting from 0 and src/dst node features of shape
    (max_num_node + 1, d).
    This reindexing is essential for the graph to be computed by a standard GNN model with PyG.
    """

    def __init__(self, device, num_nodes=None, fix_buggy_graph_reindexer=True):
        self.num_nodes = num_nodes
        self.device = device
        self.fix_buggy_graph_reindexer = fix_buggy_graph_reindexer

        self.assoc = None
        self.cache = {}
        self.is_warning = False

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None, x_is_tuple=False):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        if edge_index.min() != 0 and not self.is_warning:
            print(
                "Warning: reshaping features with non-reindexed edge index leads to large cache stored in GPU memory."
            )
            self.is_warning = True

        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1
        feature_dim = x_src.size(1)

        if feature_dim not in self.cache or self.cache[feature_dim].cache.shape[0] <= max_num_node:
            self.cache[feature_dim] = _Cache((max_num_node, feature_dim), self.device)
        self.cache[feature_dim].detach()

        # To avoid storing gradients from all nodes, we detach() BEFORE caching. If we detach()
        # after storing, we loose the gradient for all operations happening before the reindexing.
        output = self.cache[feature_dim].cache
        output.detach()
        output.zero_()

        if x_is_tuple:
            scatter(x_src, edge_index[0], out=output, dim=0, reduce="mean")
            x_src_result = output.clone()
            output.zero_()

            scatter(x_dst, edge_index[1], out=output, dim=0, reduce="mean")
            x_dst_result = output.clone()
            return (x_src_result[:max_num_node], x_dst_result[:max_num_node])
        else:
            if self.fix_buggy_graph_reindexer:
                output = output.clone()
                scatter(
                    torch.cat([x_src, x_dst]),
                    torch.cat([edge_index[0], edge_index[1]]),
                    out=output,
                    dim=0,
                    reduce="mean",
                )
            else:
                # NOTE: this one, used in orthrus and velox is buggy because it does the mean twice, which can double
                # the value of features if duplicates exist
                scatter(x_src, edge_index[0], out=output, dim=0, reduce="mean")
                scatter(x_dst, edge_index[1], out=output, dim=0, reduce="mean")

            return output[:max_num_node]

    def reindex_graph(self, data, x_is_tuple=False, use_tgn=False):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The original edge_index and node IDs are also kept.
        """
        data.original_edge_index = data.edge_index
        x, edge_index, n_id = self._reindex_graph(
            data.edge_index, data.x_src, data.x_dst, x_is_tuple=x_is_tuple
        )
        data.original_n_id = n_id
        data.x = x

        if not use_tgn:
            data.src, data.dst = edge_index[0], edge_index[1]

        data.node_type, *_ = self._reindex_graph(
            data.edge_index, data.node_type_src, data.node_type_dst, x_is_tuple=False
        )

        return data

    def _reindex_graph(
        self, edge_index, x_src=None, x_dst=None, x_is_tuple=False, max_num_node=None
    ):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.num_nodes is None:
            raise ValueError(f"Graph reindexing requires `num_nodes`.")

        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes,), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.assoc.device)
        edge_index = self.assoc[edge_index]

        if None not in [x_src, x_dst]:
            # Associates each feature vector to each reindexed node ID
            x = self.node_features_reshape(
                edge_index, x_src, x_dst, x_is_tuple=x_is_tuple, max_num_node=max_num_node
            )
        else:
            x = None

        return x, edge_index, n_id

    def to(self, device):
        self.device = device
        if self.assoc is not None:
            self.assoc = self.assoc.to(device)

        for k, v in self.cache.items():
            self.cache[k] = v.to(device)
        return self


def save_model(model, path: str, cfg):
    """
    Saves only the required weights and tensors on disk.
    Using torch.save() directly on the model is very long (up to 10min),
    so we select only the tensors we want to save/load.
    """
    os.makedirs(path, exist_ok=True)

    # We only save specific tensors, as the other tensors are not useful to save (assoc, cache, etc)
    torch.save(
        model.state_dict(),
        os.path.join(path, "state_dict.pkl"),
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    if isinstance(model.encoder, TGNEncoder):
        torch.save(
            model.encoder.neighbor_loader,
            os.path.join(path, "neighbor_loader.pkl"),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        if cfg.training.encoder.tgn.use_memory or "time_encoding" in cfg.batching.edge_features:
            torch.save(
                model.encoder.memory,
                os.path.join(path, "memory.pkl"),
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )


def load_model(model, path: str, cfg, map_location=None):
    """
    Loads weights and tensors from disk into a model.
    """
    model.load_state_dict(torch.load(os.path.join(path, "state_dict.pkl")))

    if isinstance(model.encoder, TGNEncoder):
        model.encoder.neighbor_loader = torch.load(os.path.join(path, "neighbor_loader.pkl"))
        if cfg.training.encoder.tgn.use_memory or "time_encoding" in cfg.batching.edge_features:
            model.encoder.memory = torch.load(os.path.join(path, "memory.pkl"))

    return model


def reindex_graphs(datasets, graph_reindexer, device, use_tgn, x_is_tuple=False):
    for dataset in datasets:
        for data_list in dataset:
            if isinstance(data_list, LazyBatchList):
                # Disk-backed: load each batch, reindex, save back. Only one
                # batch is resident at a time. When the list is in fp16 mode
                # we upcast wide floats to fp32 before reindexing (so the
                # internal scatter ops run at full precision and the produced
                # data.x / data.node_type match the non-lazy semantics) and
                # downcast back before saving.
                fp16 = data_list._fp16
                for i in log_tqdm(range(len(data_list)), desc="Reindexing graphs"):
                    path = data_list._path(i)
                    batch = torch.load(path, map_location="cpu")
                    if fp16:
                        _cast_lazy_keys(batch, torch.float32)
                    batch.to(device)
                    graph_reindexer.reindex_graph(
                        batch, use_tgn=use_tgn, x_is_tuple=x_is_tuple
                    )
                    batch.to("cpu")
                    if fp16:
                        _cast_lazy_keys(batch, torch.float16)
                    torch.save(batch, path)
                    batch = None
            else:
                for batch in log_tqdm(data_list, desc="Reindexing graphs"):
                    batch.to(device)
                    graph_reindexer.reindex_graph(
                        batch, use_tgn=use_tgn, x_is_tuple=x_is_tuple
                    )
                    batch.to("cpu")
