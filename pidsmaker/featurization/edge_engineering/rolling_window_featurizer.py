import math
from collections import defaultdict

import torch


CATEGORY_DIMS = {
    1: 7,
    2: 6,
    3: 4,
    4: 4,
    5: 3,
    6: 7,
    7: 3,
}


def get_engineered_feat_dim(enabled_categories):
    return sum(CATEGORY_DIMS[c] for c in enabled_categories)


def parse_enabled_categories(edge_eng_cfg):
    mapping = {
        1: getattr(edge_eng_cfg, "category_1_pair", False),
        2: getattr(edge_eng_cfg, "category_2_source", False),
        3: getattr(edge_eng_cfg, "category_3_type", False),
        4: getattr(edge_eng_cfg, "category_4_pair_type", False),
        5: getattr(edge_eng_cfg, "category_5_source_type", False),
        6: getattr(edge_eng_cfg, "category_6_global", False),
        7: getattr(edge_eng_cfg, "category_7_burst", False),
    }
    return [c for c, enabled in mapping.items() if enabled]


class RollingWindowFeatureComputer:
    """
    Computes rolling-window engineered edge features for one time-window graph.

    Processes edges in chronological order. For each edge, features are read
    from the current state (before insertion), then the state is updated.
    This guarantees causality: no feature leaks information from the current
    edge or future edges.

    Feature categories (1-7) are individually toggleable via config.

    Performance notes:
      - _cat6_global avg_fan_out is O(1) per edge via running _total_fan_out counter.
      - _cat1_pair IAT statistics use Welford's online algorithm (O(1) per edge
        instead of O(k) recomputation from scratch).
      - No unbounded timestamp lists: pair_timestamps is replaced by lightweight
        running counters (pair_count, pair_first_ts, pair_last_ts + IAT stats).
    """

    def __init__(self, enabled_categories, window_duration_ns, num_edge_types):
        self.enabled = sorted(enabled_categories)
        self.window_duration_ns = window_duration_ns
        self.window_duration_s = window_duration_ns / 1e9
        self.num_edge_types = num_edge_types
        self.out_dim = get_engineered_feat_dim(self.enabled)
        self.reset()

    def reset(self):
        # --- pair-level state (replaces pair_timestamps list) ---
        self.pair_count = defaultdict(int)
        self.pair_first_ts = {}
        self.pair_last_ts = {}
        self.pair_iat_mean = defaultdict(float)
        self.pair_iat_m2 = defaultdict(float)
        self.pair_iat_n = defaultdict(int)

        # --- source-level state ---
        self.src_count = defaultdict(int)
        self.src_last_seen = {}
        self.src_unique_dst = defaultdict(set)
        self.src_unique_types = defaultdict(set)

        # --- per-edge-type state ---
        self.type_count = defaultdict(int)
        self.type_last_seen = {}

        # --- (src, dst, type) triple state ---
        self.triple_count = defaultdict(int)
        self.triple_last_seen = {}

        # --- (src, type) state ---
        self.src_type_count = defaultdict(int)
        self.src_type_last_seen = {}

        # --- global window state ---
        self.window_total = 0
        self.window_src_set = set()
        self.window_dst_set = set()
        self.window_pair_set = set()
        self.window_type_set = set()

        # running sum of src fan-outs (for O(1) avg_fan_out in cat6)
        self._total_fan_out = 0

    def compute(self, edges):
        """
        edges: list of (src, dst, edge_type_idx, timestamp_ns) sorted by time.
        edge_type_idx is an integer index (from rel2id).
        Returns: (E, out_dim) tensor of engineered features.
        """
        all_feats = []
        for src, dst, etype_idx, ts in edges:
            feats = self._read_features(src, dst, etype_idx, ts)
            self._update_state(src, dst, etype_idx, ts)
            all_feats.append(feats)
        return torch.tensor(all_feats, dtype=torch.float)

    def _read_features(self, src, dst, etype_idx, ts):
        feats = []
        pair_key = (src, dst)
        triple_key = (src, dst, etype_idx)
        src_type_key = (src, etype_idx)

        for cat in self.enabled:
            if cat == 1:
                feats.extend(self._cat1_pair(pair_key, ts))
            elif cat == 2:
                feats.extend(self._cat2_source(src, ts))
            elif cat == 3:
                feats.extend(self._cat3_type(etype_idx, ts))
            elif cat == 4:
                feats.extend(self._cat4_pair_type(triple_key, pair_key, etype_idx, ts))
            elif cat == 5:
                feats.extend(self._cat5_source_type(src_type_key, src, ts))
            elif cat == 6:
                feats.extend(self._cat6_global(ts))
            elif cat == 7:
                feats.extend(self._cat7_burst(pair_key, src, triple_key))
        return feats

    def _update_state(self, src, dst, etype_idx, ts):
        pair_key = (src, dst)
        triple_key = (src, dst, etype_idx)
        src_type_key = (src, etype_idx)

        # --- pair IAT running stats (Welford, before updating count/last) ---
        if pair_key in self.pair_last_ts:
            prev_ts = self.pair_last_ts[pair_key]
            iat = ts - prev_ts
            self.pair_iat_n[pair_key] += 1
            n = self.pair_iat_n[pair_key]
            mean = self.pair_iat_mean[pair_key]
            delta = iat - mean
            self.pair_iat_mean[pair_key] = mean + delta / n
            self.pair_iat_m2[pair_key] += delta * (iat - self.pair_iat_mean[pair_key])
        else:
            self.pair_first_ts[pair_key] = ts

        self.pair_last_ts[pair_key] = ts
        self.pair_count[pair_key] += 1

        # --- running total_fan_out (for O(1) avg_fan_out in cat6) ---
        if dst not in self.src_unique_dst[src]:
            self._total_fan_out += 1

        # --- source state ---
        self.src_count[src] += 1
        self.src_last_seen[src] = ts
        self.src_unique_dst[src].add(dst)
        self.src_unique_types[src].add(etype_idx)

        # --- type state ---
        self.type_count[etype_idx] += 1
        self.type_last_seen[etype_idx] = ts

        # --- triple state ---
        self.triple_count[triple_key] += 1
        self.triple_last_seen[triple_key] = ts

        # --- src-type state ---
        self.src_type_count[src_type_key] += 1
        self.src_type_last_seen[src_type_key] = ts

        # --- global window ---
        self.window_total += 1
        self.window_src_set.add(src)
        self.window_dst_set.add(dst)
        self.window_pair_set.add(pair_key)
        self.window_type_set.add(etype_idx)

    # --- Category helpers ---

    def _time_ratio(self, delta_ns):
        if delta_ns is None:
            return 1.0
        return min(delta_ns / self.window_duration_ns, 1.0)

    def _cat1_pair(self, pair_key, ts):
        count = self.pair_count.get(pair_key, 0)
        last_ts = self.pair_last_ts.get(pair_key)
        first_ts = self.pair_first_ts.get(pair_key)

        if count == 0:
            time_since_last = 1.0
            first_seen_offset = 0.0
            span = 0.0
            iat_mean = 0.0
            iat_std = 0.0
        else:
            time_since_last = self._time_ratio(ts - last_ts)
            first_seen_offset = self._time_ratio(ts - first_ts)
            span = self._time_ratio(last_ts - first_ts) if count > 1 else 0.0

            iat_n = self.pair_iat_n.get(pair_key, 0)
            if iat_n >= 1:
                iat_mean = self.pair_iat_mean[pair_key] / self.window_duration_ns
                iat_std = (
                    math.sqrt(self.pair_iat_m2[pair_key] / iat_n) / self.window_duration_ns
                    if iat_n >= 2
                    else 0.0
                )
            else:
                iat_mean = 0.0
                iat_std = 0.0

        rate = count / self.window_duration_s if self.window_duration_s > 0 else 0.0

        return [count, time_since_last, first_seen_offset, span, iat_mean, iat_std, rate]

    def _cat2_source(self, src, ts):
        count = self.src_count.get(src, 0)
        unique_dst = len(self.src_unique_dst.get(src, set()))
        unique_types = len(self.src_unique_types.get(src, set()))
        last_ts = self.src_last_seen.get(src)
        time_since_last = self._time_ratio(ts - last_ts) if last_ts is not None else 1.0
        rate = count / self.window_duration_s if self.window_duration_s > 0 else 0.0
        diversity = unique_dst / max(count, 1)

        return [count, unique_dst, unique_types, time_since_last, rate, diversity]

    def _cat3_type(self, etype_idx, ts):
        count = self.type_count.get(etype_idx, 0)
        ratio = count / max(self.window_total, 1)
        last_ts = self.type_last_seen.get(etype_idx)
        time_since_last = self._time_ratio(ts - last_ts) if last_ts is not None else 1.0
        rate = count / self.window_duration_s if self.window_duration_s > 0 else 0.0

        return [count, ratio, time_since_last, rate]

    def _cat4_pair_type(self, triple_key, pair_key, etype_idx, ts):
        count = self.triple_count.get(triple_key, 0)
        last_ts = self.triple_last_seen.get(triple_key)
        time_since_last = self._time_ratio(ts - last_ts) if last_ts is not None else 1.0
        pair_count = self.pair_count.get(pair_key, 0)
        ratio_in_pair = count / max(pair_count, 1)
        type_count = self.type_count.get(etype_idx, 0)
        ratio_in_type = count / max(type_count, 1)

        return [count, time_since_last, ratio_in_pair, ratio_in_type]

    def _cat5_source_type(self, src_type_key, src, ts):
        count = self.src_type_count.get(src_type_key, 0)
        last_ts = self.src_type_last_seen.get(src_type_key)
        time_since_last = self._time_ratio(ts - last_ts) if last_ts is not None else 1.0
        src_total = self.src_count.get(src, 0)
        ratio = count / max(src_total, 1)

        return [count, time_since_last, ratio]

    def _cat6_global(self, ts):
        total = self.window_total
        unique_src = len(self.window_src_set)
        unique_dst = len(self.window_dst_set)
        unique_pairs = len(self.window_pair_set)
        unique_types = len(self.window_type_set)

        if total > 1 and unique_types > 0:
            entropy = 0.0
            for etype_idx in self.window_type_set:
                p = self.type_count.get(etype_idx, 0) / total
                if p > 0:
                    entropy -= p * math.log(p)
            max_entropy = math.log(max(unique_types, 1))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            norm_entropy = 0.0

        avg_fan_out = (
            self._total_fan_out / max(len(self.window_src_set), 1)
            if self.window_src_set
            else 0.0
        )

        return [total, unique_src, unique_dst, unique_pairs, unique_types, norm_entropy, avg_fan_out]

    def _cat7_burst(self, pair_key, src, triple_key):
        total = max(self.window_total, 1)
        pair_count = self.pair_count.get(pair_key, 0)
        src_count = self.src_count.get(src, 0)
        triple_count = self.triple_count.get(triple_key, 0)

        pair_burst = pair_count / total
        src_burst = src_count / total
        triple_dominance = triple_count / total

        return [pair_burst, src_burst, triple_dominance]
