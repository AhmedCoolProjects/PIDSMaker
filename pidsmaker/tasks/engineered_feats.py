"""Rolling-window statistical edge features for VELOX.

Computes 9 causal features per edge in O(1) using incremental state:
  pair_recurrence, src_fanout, dst_fanin, src_op_freq, global_op_freq,
  time_since_src_last, time_since_pair_last, src_ema_rate, src_op_entropy.
"""

import math
from collections import defaultdict

import torch

ENGINEERED_FEAT_DIM = 9

# Index constants for the 9-dim feature vector
F_PAIR_RECURRENCE = 0
F_SRC_FANOUT = 1
F_DST_FANIN = 2
F_SRC_OP_FREQ = 3
F_GLOBAL_OP_FREQ = 4
F_TIME_SINCE_SRC = 5
F_TIME_SINCE_PAIR = 6
F_SRC_EMA_RATE = 7
F_SRC_ENTROPY = 8


class EngineeredFeatureComputer:
    """Per-window incremental computer for engineered edge features.

    State resets on init (one per time window).  Call ``step(src, dst, op_type, timestamp_ns)``
    for each edge in sorted order — features are computed causally from edges *before* the
    current one, then state is updated.

    Parameters
    ----------
    num_op_types:
        Number of distinct operation types (varies per dataset, 8–25).
    ema_alpha:
        Decay factor for the burstiness EMA (default 0.05).
    use_log1p:
        If True, apply ``log(1+x)`` to count and delta-time features.
    enabled_families:
        Set of feature family names to keep; all others are zeroed out.
        Accepts any subset of ``{"pair_recurrence", "source_fanout", "dst_fanin",
        "op_rarity", "temporal", "burstiness", "entropy"}``.
        If None, all families are enabled.
    """

    def __init__(
        self,
        num_op_types: int,
        ema_alpha: float = 0.05,
        use_log1p: bool = True,
        enabled_families=None,
    ):
        self.num_op_types = num_op_types
        self.ema_alpha = ema_alpha
        self.use_log1p = use_log1p

        if enabled_families is None:
            enabled_families = {
                "pair_recurrence", "source_fanout", "dst_fanin",
                "op_rarity", "temporal", "burstiness", "entropy",
            }
        self._enabled = enabled_families

        self.pair_counts: dict[tuple[int, int], int] = {}
        self.src_fanout: dict[int, set[int]] = defaultdict(set)
        self.dst_fanin: dict[int, set[int]] = defaultdict(set)
        self.src_op_counts: dict[int, list[int]] = {}
        self.global_op_counts: list[int] = [0] * num_op_types
        self.src_total_ops: dict[int, int] = {}
        self.global_total_ops: int = 0
        self.last_src_time: dict[int, int] = {}
        self.last_pair_time: dict[tuple[int, int], int] = {}
        self.src_ema_rate: dict[int, float] = {}
        self.src_last_ema_time: dict[int, int] = {}

    def _log1p(self, x: float) -> float:
        return math.log1p(x) if self.use_log1p else x

    def step(self, src: int, dst: int, op_type: int, timestamp_ns: int) -> torch.Tensor:
        """Return 9-d causal feature vector for this edge, then update state."""
        feats = [0.0] * ENGINEERED_FEAT_DIM
        pair = (src, dst)

        if "pair_recurrence" in self._enabled:
            pair_count = self.pair_counts.get(pair, 0)
            feats[F_PAIR_RECURRENCE] = self._log1p(float(pair_count))
        else:
            pair_count = 0

        if "source_fanout" in self._enabled:
            src_fanout_count = len(self.src_fanout.get(src, set()))
            feats[F_SRC_FANOUT] = self._log1p(float(src_fanout_count))
        else:
            src_fanout_count = 0

        if "dst_fanin" in self._enabled:
            dst_fanin_count = len(self.dst_fanin.get(dst, set()))
            feats[F_DST_FANIN] = self._log1p(float(dst_fanin_count))
        else:
            dst_fanin_count = 0

        if "op_rarity" in self._enabled:
            src_total = self.src_total_ops.get(src, 0)
            feats[F_SRC_OP_FREQ] = (
                (self.src_op_counts[src][op_type] / src_total)
                if src in self.src_op_counts and src_total > 0
                else 0.0
            )
            feats[F_GLOBAL_OP_FREQ] = (
                (self.global_op_counts[op_type] / self.global_total_ops)
                if self.global_total_ops > 0
                else 0.0
            )

        if "temporal" in self._enabled:
            last_src_ts = self.last_src_time.get(src)
            time_since_src_ns = (timestamp_ns - last_src_ts) if last_src_ts is not None else 0
            last_pair_ts = self.last_pair_time.get(pair)
            time_since_pair_ns = (timestamp_ns - last_pair_ts) if last_pair_ts is not None else 0
            time_since_src_sec = max(0, time_since_src_ns / 1_000_000_000.0)
            time_since_pair_sec = max(0, time_since_pair_ns / 1_000_000_000.0)
            feats[F_TIME_SINCE_SRC] = (
                self._log1p(time_since_src_sec) if time_since_src_sec > 0 else 0.0
            )
            feats[F_TIME_SINCE_PAIR] = (
                self._log1p(time_since_pair_sec) if time_since_pair_sec > 0 else 0.0
            )

        prev_ema = self.src_ema_rate.get(src, 0.0)
        if "burstiness" in self._enabled:
            feats[F_SRC_EMA_RATE] = prev_ema

        if "entropy" in self._enabled and src in self.src_op_counts:
            counts = self.src_op_counts[src]
            total = self.src_total_ops[src]
            entropy = 0.0
            for c in counts:
                if c > 0:
                    p = c / total
                    entropy -= p * math.log2(p)
            norm = math.log2(self.num_op_types) if self.num_op_types > 1 else 1.0
            feats[F_SRC_ENTROPY] = entropy / norm

        # --- update state after reading ---
        if "pair_recurrence" in self._enabled:
            self.pair_counts[pair] = pair_count + 1
        if "source_fanout" in self._enabled:
            self.src_fanout[src].add(dst)
        if "dst_fanin" in self._enabled:
            self.dst_fanin[dst].add(src)

        if "op_rarity" in self._enabled or "entropy" in self._enabled:
            if src not in self.src_op_counts:
                self.src_op_counts[src] = [0] * self.num_op_types
                self.src_total_ops[src] = 0
            self.src_op_counts[src][op_type] += 1
            self.src_total_ops[src] += 1
            self.global_op_counts[op_type] += 1
            self.global_total_ops += 1

        if "burstiness" in self._enabled or "temporal" in self._enabled:
            last_call_ts = self.last_src_time.get(src)
            if last_call_ts is not None and timestamp_ns > last_call_ts:
                dt_sec = max(1.0, (timestamp_ns - last_call_ts) / 1_000_000_000.0)
                rate = 1.0 / dt_sec
                self.src_ema_rate[src] = self.ema_alpha * rate + (1.0 - self.ema_alpha) * prev_ema
            else:
                self.src_ema_rate[src] = prev_ema
            self.last_src_time[src] = timestamp_ns

        if "temporal" in self._enabled or "pair_recurrence" in self._enabled:
            self.last_pair_time[pair] = timestamp_ns

        feats = [max(-1e3, min(1e3, v)) for v in feats]
        return torch.tensor(feats, dtype=torch.float)
