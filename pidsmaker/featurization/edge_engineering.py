"""Causal, O(1)-per-edge rolling-window edge feature engineering for VELOX.

Builds a per-edge feature vector that captures behavioral context within the
current time window: pair recurrence, source fan-out, destination fan-in,
operation rarity for the source, inter-arrival times, burstiness EMA, and
operation-type entropy. All families are toggleable from config.

State is window-scoped and reset between graphs (the caller is responsible for
instantiating a fresh ``EdgeFeatureBuilder`` per window). The emitter is
strictly causal: for edge ``i`` the returned vector depends only on edges
``0..i-1``; state is updated only after the vector is emitted.
"""

import math
from collections import defaultdict
from typing import Iterable, Optional

import torch


FAMILIES = (
    "pair_recurrence",
    "src_fanout",
    "dst_fanin",
    "op_rarity",
    "temporal",
    "burstiness",
    "type_mix_entropy",
)


def _family_dims(num_op_types: int):
    return {
        "pair_recurrence": 2,   # [log1p(prev_pair_count), is_first_seen]
        "src_fanout": 1,        # [log1p(|unique_dsts(src)|)]
        "dst_fanin": 1,         # [log1p(|unique_srcs(dst)|)]
        "op_rarity": 3,         # [log1p(prev_op_count_for_src), cond_freq, novelty]
        "temporal": 4,          # [log1p_dt_src, has_prev_src, log1p_dt_pair, has_prev_pair]
        "burstiness": 2,        # [ema_rate_src, log1p(events_so_far_src)]
        "type_mix_entropy": 1,  # [H(op|src) / log(num_op_types)]
    }


def parse_families(families) -> list[str]:
    """Normalise a families spec (str | iterable) to a deduped, validated list."""
    if isinstance(families, str):
        items = [f.strip() for f in families.replace("-", ",").split(",") if f.strip()]
    else:
        items = [str(f).strip() for f in families]
    if not items or items == ["none"]:
        return []
    bad = [f for f in items if f not in FAMILIES]
    if bad:
        raise ValueError(
            f"Unknown engineered feature family(s) {bad}. "
            f"Valid options: {list(FAMILIES)}"
        )
    # Preserve order, dedup.
    seen, out = set(), []
    for f in items:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def compute_engineered_dim(families, num_op_types: int) -> int:
    """Dimension of the engineered feature vector for a given families config."""
    fam = parse_families(families)
    dims = _family_dims(num_op_types)
    return sum(dims[f] for f in fam)


class EdgeFeatureBuilder:
    """Per-window incremental state and O(1)-per-edge feature emitter."""

    def __init__(
        self,
        num_op_types: int,
        families: Iterable[str],
        ema_alpha: float = 0.3,
        log1p_counts: bool = True,
        standardize_delta_t: bool = False,
    ):
        self.num_op_types = max(int(num_op_types), 1)
        self.families = parse_families(families)
        self.ema_alpha = float(ema_alpha)
        self.log1p_counts = bool(log1p_counts)
        self.standardize_delta_t = bool(standardize_delta_t)

        self._dims = _family_dims(self.num_op_types)
        self.feat_dim = sum(self._dims[f] for f in self.families)

        # Pre-compute the column slice (start, end) per active family inside the
        # emitted vector. Used by finalize() for selective standardisation.
        self._slices = {}
        off = 0
        for f in self.families:
            self._slices[f] = (off, off + self._dims[f])
            off += self._dims[f]

        # ---- Window-scoped state ----
        self.pair_count: dict = defaultdict(int)
        self.pair_last_t: dict = {}
        self.src_last_t: dict = {}
        self.src_fanout: dict = defaultdict(set)
        self.dst_fanin: dict = defaultdict(set)
        self.src_op_counts: dict = defaultdict(lambda: defaultdict(int))
        self.src_op_total: dict = defaultdict(int)
        self.src_ema_rate: dict = {}
        # Incremental support for Shannon entropy of op distribution per src:
        #   H = log(N) - (1/N) * sum_i c_i * log(c_i)
        # We track ``sum_c_log_c`` and ``N=src_op_total[src]``.
        self.src_sum_c_log_c: dict = defaultdict(float)

    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_log1p(x: float, do_log1p: bool) -> float:
        if not do_log1p:
            return float(x)
        if x < 0:
            x = 0.0
        return math.log1p(x)

    def emit(self, src: int, dst: int, op_id: Optional[int], t: int) -> list[float]:
        """Return the engineered feature vector for the current edge.

        State is *not* yet updated when this returns — call :meth:`update`
        right after (or use :meth:`emit_and_update`).
        """
        feats: list[float] = []
        log1p_counts = self.log1p_counts

        if "pair_recurrence" in self._slices:
            cnt = self.pair_count.get((src, dst), 0)
            feats.append(self._maybe_log1p(cnt, log1p_counts))
            feats.append(1.0 if cnt == 0 else 0.0)

        if "src_fanout" in self._slices:
            feats.append(self._maybe_log1p(len(self.src_fanout.get(src, ())), log1p_counts))

        if "dst_fanin" in self._slices:
            feats.append(self._maybe_log1p(len(self.dst_fanin.get(dst, ())), log1p_counts))

        if "op_rarity" in self._slices:
            if op_id is None:
                feats.extend([0.0, 0.0, 1.0])
            else:
                count_op = self.src_op_counts.get(src, {}).get(op_id, 0)
                total_op = self.src_op_total.get(src, 0)
                cond = (count_op / total_op) if total_op > 0 else 0.0
                novelty = 1.0 if count_op == 0 else 0.0
                feats.extend([self._maybe_log1p(count_op, log1p_counts), cond, novelty])

        if "temporal" in self._slices:
            prev_src = self.src_last_t.get(src)
            prev_pair = self.pair_last_t.get((src, dst))
            dt_src = float(t - prev_src) if prev_src is not None else 0.0
            dt_pair = float(t - prev_pair) if prev_pair is not None else 0.0
            has_prev_src = 1.0 if prev_src is not None else 0.0
            has_prev_pair = 1.0 if prev_pair is not None else 0.0
            # Δt is always log1p'd because it spans nanosecond → minutes ranges.
            feats.append(math.log1p(max(dt_src, 0.0)))
            feats.append(has_prev_src)
            feats.append(math.log1p(max(dt_pair, 0.0)))
            feats.append(has_prev_pair)

        if "burstiness" in self._slices:
            feats.append(float(self.src_ema_rate.get(src, 0.0)))
            feats.append(self._maybe_log1p(self.src_op_total.get(src, 0), log1p_counts))

        if "type_mix_entropy" in self._slices:
            n = self.src_op_total.get(src, 0)
            if n > 0:
                h = math.log(n) - (self.src_sum_c_log_c.get(src, 0.0) / n)
                denom = math.log(self.num_op_types) if self.num_op_types > 1 else 1.0
                feats.append(h / denom)
            else:
                feats.append(0.0)

        return feats

    # ------------------------------------------------------------------
    def update(self, src: int, dst: int, op_id: Optional[int], t: int) -> None:
        """Fold this edge into the running state. Always call after :meth:`emit`."""
        # Burstiness EMA: rate-per-second of this src, before recording t.
        prev_t = self.src_last_t.get(src)
        if prev_t is not None:
            dt_ns = max(t - prev_t, 1)
            instant_rate = 1e9 / dt_ns  # events/sec
            ema_prev = self.src_ema_rate.get(src, instant_rate)
            self.src_ema_rate[src] = (
                self.ema_alpha * instant_rate + (1.0 - self.ema_alpha) * ema_prev
            )
        else:
            # First sighting — EMA stays at 0 until we have a Δt.
            self.src_ema_rate.setdefault(src, 0.0)

        self.src_last_t[src] = t
        self.pair_last_t[(src, dst)] = t
        self.pair_count[(src, dst)] += 1
        self.src_fanout[src].add(dst)
        self.dst_fanin[dst].add(src)

        if op_id is not None:
            counts = self.src_op_counts[src]
            old_c = counts.get(op_id, 0)
            new_c = old_c + 1
            counts[op_id] = new_c
            old_term = old_c * math.log(old_c) if old_c > 0 else 0.0
            new_term = new_c * math.log(new_c)
            self.src_sum_c_log_c[src] += (new_term - old_term)
            self.src_op_total[src] += 1

    def emit_and_update(
        self, src: int, dst: int, op_id: Optional[int], t: int
    ) -> list[float]:
        v = self.emit(src, dst, op_id, t)
        self.update(src, dst, op_id, t)
        return v

    # ------------------------------------------------------------------
    def finalize(self, feats: torch.Tensor) -> torch.Tensor:
        """Optional per-window post-processing (z-score Δt columns).

        Operates on the assembled (E, feat_dim) tensor. No-op unless
        ``standardize_delta_t=True`` and the ``temporal`` family is enabled.
        """
        if feats.numel() == 0 or not self.standardize_delta_t:
            return feats
        if "temporal" not in self._slices:
            return feats

        start, _ = self._slices["temporal"]
        # Layout inside temporal: [log1p_dt_src, has_prev_src, log1p_dt_pair, has_prev_pair]
        dt_src_col, has_src_col = start + 0, start + 1
        dt_pair_col, has_pair_col = start + 2, start + 3

        for dt_col, has_col in ((dt_src_col, has_src_col), (dt_pair_col, has_pair_col)):
            mask = feats[:, has_col] > 0.5
            if mask.sum() < 2:
                continue
            vals = feats[mask, dt_col]
            mean = vals.mean()
            std = vals.std(unbiased=False).clamp_min(1e-6)
            feats[mask, dt_col] = (vals - mean) / std

        return feats
