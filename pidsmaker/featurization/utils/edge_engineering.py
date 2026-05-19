from __future__ import annotations

import math
from collections import defaultdict

import torch


class RunningStandardizer:
    """Causal running standardization with Welford statistics."""

    def __init__(self, eps: float = 1e-6):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def transform(self, value: float) -> float:
        if self.count < 2:
            return 0.0
        var = self.m2 / max(self.count - 1, 1)
        return (value - self.mean) / math.sqrt(var + self.eps)

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2


def _cfg_value(cfg, path: list[str], default):
    """Read config value from either root (`cfg.edge_engineering.*`) or
    nested (`cfg.feat_inference.edge_engineering.*`) locations."""

    candidates = [cfg]
    feat_inference_cfg = getattr(cfg, "feat_inference", None)
    if feat_inference_cfg is not None:
        candidates.append(feat_inference_cfg)

    for candidate in candidates:
        obj = candidate
        found = True
        for key in path:
            obj = getattr(obj, key, None)
            if obj is None:
                found = False
                break
        if found:
            return obj

    return default


def edge_engineering_enabled(cfg) -> bool:
    return bool(_cfg_value(cfg, ["edge_engineering", "enable"], False))


def get_engineered_feat_dim(cfg) -> int:
    if not edge_engineering_enabled(cfg):
        return 0

    dim = 0
    if _cfg_value(cfg, ["edge_engineering", "pair_recurrence", "enabled"], True):
        dim += 3
    if _cfg_value(cfg, ["edge_engineering", "fan", "enabled"], True):
        dim += 4
    if _cfg_value(cfg, ["edge_engineering", "operation_rarity", "enabled"], True):
        dim += 3
    if _cfg_value(cfg, ["edge_engineering", "temporal", "enabled"], True):
        dim += 3
    if _cfg_value(cfg, ["edge_engineering", "burstiness", "enabled"], True):
        dim += 2
    if _cfg_value(cfg, ["edge_engineering", "op_mix", "enabled"], True):
        dim += 2
    return dim


class EngineeredEdgeFeatureBuilder:
    """Build causal O(1)-update engineered edge features for one graph window."""

    def __init__(self, cfg, rel2id: dict):
        self.cfg = cfg
        self.log1p_counts = bool(_cfg_value(cfg, ["edge_engineering", "log1p_counts"], True))
        self.delta_time_unit = str(
            _cfg_value(cfg, ["edge_engineering", "delta_time_unit"], "seconds")
        )
        self.eps = float(_cfg_value(cfg, ["edge_engineering", "eps"], 1e-6))

        self.use_pair = bool(
            _cfg_value(cfg, ["edge_engineering", "pair_recurrence", "enabled"], True)
        )
        self.use_fan = bool(_cfg_value(cfg, ["edge_engineering", "fan", "enabled"], True))
        self.use_op_rarity = bool(
            _cfg_value(cfg, ["edge_engineering", "operation_rarity", "enabled"], True)
        )
        self.use_temporal = bool(_cfg_value(cfg, ["edge_engineering", "temporal", "enabled"], True))
        self.use_burst = bool(_cfg_value(cfg, ["edge_engineering", "burstiness", "enabled"], True))
        self.use_op_mix = bool(_cfg_value(cfg, ["edge_engineering", "op_mix", "enabled"], True))

        self.temporal_standardize = bool(
            _cfg_value(cfg, ["edge_engineering", "temporal", "standardize_delta_t"], False)
        )
        self.ema_alpha = float(
            _cfg_value(cfg, ["edge_engineering", "burstiness", "ema_alpha"], 0.2)
        )

        self.rel_label_to_id = {k: int(v) - 1 for k, v in rel2id.items() if isinstance(k, str)}

        # Stateful counters / maps (reset once per graph window)
        self.pair_count = defaultdict(int)
        self.last_pair_t = {}

        self.src_unique_dsts = defaultdict(set)
        self.dst_unique_srcs = defaultdict(set)
        self.src_total = defaultdict(int)
        self.dst_total = defaultdict(int)

        self.global_op_count = defaultdict(int)
        self.src_op_count = defaultdict(int)

        self.last_src_t = {}
        self.last_dst_t = {}
        self.last_global_t = None

        self.src_rate_ema = defaultdict(float)

        self.src_sum_sq = defaultdict(float)
        self.src_unique_ops = defaultdict(set)

        self.std_pair_dt = RunningStandardizer(eps=self.eps)
        self.std_src_dt = RunningStandardizer(eps=self.eps)
        self.std_dst_dt = RunningStandardizer(eps=self.eps)
        self.std_global_dt = RunningStandardizer(eps=self.eps)

    def _count_transform(self, value: int) -> float:
        if self.log1p_counts:
            return math.log1p(float(value))
        return float(value)

    def _dt_to_seconds(self, dt_ns: int) -> float:
        if self.delta_time_unit == "milliseconds":
            return float(dt_ns) / 1_000_000.0
        return float(dt_ns) / 1_000_000_000.0

    def _delta_t(self, last_t: int | None, curr_t: int) -> float | None:
        if last_t is None:
            return None
        dt_ns = max(int(curr_t) - int(last_t), 0)
        return self._dt_to_seconds(dt_ns)

    def _temporal_feature(self, delta_t: float | None, standardizer: RunningStandardizer) -> float:
        if delta_t is None:
            return 0.0
        if self.temporal_standardize:
            return standardizer.transform(delta_t)
        return math.log1p(delta_t)

    def _update_temporal_standardizer(
        self, delta_t: float | None, standardizer: RunningStandardizer
    ):
        if delta_t is not None and self.temporal_standardize:
            standardizer.update(delta_t)

    def _resolve_op_idx(self, edge_label: str | None) -> int:
        if edge_label is None:
            return -1
        return int(self.rel_label_to_id.get(edge_label, -1))

    def compute_edge_features(
        self, src: int, dst: int, timestamp: int, edge_label: str | None
    ) -> list[float]:
        op_idx = self._resolve_op_idx(edge_label)
        feats: list[float] = []

        pair_key = (src, dst)
        src_total_prev = self.src_total[src]
        dst_total_prev = self.dst_total[dst]

        pair_count_prev = self.pair_count[pair_key]
        pair_seen_prev = 1.0 if pair_count_prev > 0 else 0.0
        pair_dt = self._delta_t(self.last_pair_t.get(pair_key), timestamp)

        src_dt = self._delta_t(self.last_src_t.get(src), timestamp)
        dst_dt = self._delta_t(self.last_dst_t.get(dst), timestamp)
        global_dt = self._delta_t(self.last_global_t, timestamp)

        src_rate_ema_prev = self.src_rate_ema[src]
        instant_src_rate = 0.0 if src_dt is None else 1.0 / (src_dt + self.eps)
        src_rate_ratio = (
            instant_src_rate / (src_rate_ema_prev + self.eps) if src_rate_ema_prev > 0.0 else 0.0
        )

        src_unique_dst_prev = len(self.src_unique_dsts[src])
        dst_unique_src_prev = len(self.dst_unique_srcs[dst])

        if op_idx >= 0:
            global_op_count_prev = self.global_op_count[op_idx]
            src_op_count_prev = self.src_op_count[(src, op_idx)]
        else:
            global_op_count_prev = 0
            src_op_count_prev = 0

        src_op_ratio_prev = (src_op_count_prev / src_total_prev) if src_total_prev > 0 else 0.0

        if src_total_prev > 0:
            concentration_prev = self.src_sum_sq[src] / max(
                float(src_total_prev * src_total_prev), 1.0
            )
        else:
            concentration_prev = 0.0
        unique_ops_prev = len(self.src_unique_ops[src])

        if self.use_pair:
            feats.extend(
                [
                    self._count_transform(pair_count_prev),
                    pair_seen_prev,
                    self._temporal_feature(pair_dt, self.std_pair_dt),
                ]
            )

        if self.use_fan:
            feats.extend(
                [
                    self._count_transform(src_unique_dst_prev),
                    self._count_transform(src_total_prev),
                    self._count_transform(dst_unique_src_prev),
                    self._count_transform(dst_total_prev),
                ]
            )

        if self.use_op_rarity:
            feats.extend(
                [
                    self._count_transform(global_op_count_prev),
                    self._count_transform(src_op_count_prev),
                    float(src_op_ratio_prev),
                ]
            )

        if self.use_temporal:
            feats.extend(
                [
                    self._temporal_feature(src_dt, self.std_src_dt),
                    self._temporal_feature(dst_dt, self.std_dst_dt),
                    self._temporal_feature(global_dt, self.std_global_dt),
                ]
            )

        if self.use_burst:
            feats.extend([float(src_rate_ema_prev), float(src_rate_ratio)])

        if self.use_op_mix:
            feats.extend([float(1.0 - concentration_prev), self._count_transform(unique_ops_prev)])

        # Update states after reading current features (causal)
        self.pair_count[pair_key] = pair_count_prev + 1
        self.last_pair_t[pair_key] = timestamp

        self.src_unique_dsts[src].add(dst)
        self.dst_unique_srcs[dst].add(src)
        self.src_total[src] = src_total_prev + 1
        self.dst_total[dst] = dst_total_prev + 1

        if op_idx >= 0:
            self.global_op_count[op_idx] = global_op_count_prev + 1

            src_op_key = (src, op_idx)
            prev_op_count = self.src_op_count[src_op_key]
            self.src_op_count[src_op_key] = prev_op_count + 1

            self.src_sum_sq[src] += 2.0 * prev_op_count + 1.0
            self.src_unique_ops[src].add(op_idx)

        self.last_src_t[src] = timestamp
        self.last_dst_t[dst] = timestamp
        self.last_global_t = timestamp

        self.src_rate_ema[src] = (
            self.ema_alpha * instant_src_rate + (1.0 - self.ema_alpha) * src_rate_ema_prev
        )

        self._update_temporal_standardizer(pair_dt, self.std_pair_dt)
        self._update_temporal_standardizer(src_dt, self.std_src_dt)
        self._update_temporal_standardizer(dst_dt, self.std_dst_dt)
        self._update_temporal_standardizer(global_dt, self.std_global_dt)

        return feats


def compute_engineered_feats_for_edges(cfg, rel2id, sorted_edges) -> torch.Tensor | None:
    if not edge_engineering_enabled(cfg):
        return None

    builder = EngineeredEdgeFeatureBuilder(cfg=cfg, rel2id=rel2id)
    feat_dim = get_engineered_feat_dim(cfg)
    if feat_dim == 0:
        return None

    engineered_rows = []
    for u, v, _, attr in sorted_edges:
        src = int(u)
        dst = int(v)
        timestamp = int(attr["time"])
        edge_label = attr.get("label", None)
        engineered_rows.append(builder.compute_edge_features(src, dst, timestamp, edge_label))

    return torch.tensor(engineered_rows, dtype=torch.float)
