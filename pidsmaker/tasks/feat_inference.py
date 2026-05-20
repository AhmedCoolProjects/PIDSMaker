import os

import torch

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.featurization.feat_inference_methods import (
    feat_inference_alacarte,
    feat_inference_doc2vec,
    feat_inference_fasttext,
    feat_inference_flash,
    feat_inference_HFH,
    feat_inference_TRW,
    feat_inference_word2vec,
)
from pidsmaker.utils.data_utils import CollatableTemporalData
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


import math
from collections import defaultdict

def compute_engineered_feats(graph, cfg):
    raw_edges = graph.graph.get("raw_edges", None)
    fusion_idxs = graph.graph.get("fusion_idxs", None)
    sorted_edges = graph.edges(data=True, keys=True)

    if raw_edges is None or fusion_idxs is None:
        # Reconstruct chronologically sorted raw edges
        raw_edges = []
        for u, v, k, attr in sorted(sorted_edges, key=lambda x: x[3].get("time", 0)):
            raw_edges.append({
                "src": int(u),
                "dst": int(v),
                "time": int(attr.get("time", 0)),
                "label": attr.get("label", "unknown"),
                "event_uuid": attr.get("event_uuid", "")
            })
        
        # Build mapping from raw edges to their index in sorted_edges
        sorted_edges_list = list(sorted_edges)
        edge_to_fused_idx = {}
        for f_idx, (u, v, k, attr) in enumerate(sorted_edges_list):
            edge_to_fused_idx[(int(u), int(v), int(attr.get("time", 0)), attr.get("label", "unknown"))] = f_idx
            
        fusion_idxs = []
        for raw_edge in raw_edges:
            key = (raw_edge["src"], raw_edge["dst"], raw_edge["time"], raw_edge["label"])
            fused_idx = edge_to_fused_idx.get(key, 0)
            fusion_idxs.append(fused_idx)

    E_raw = len(raw_edges)
    E_fused = len(sorted_edges)
    
    ee = getattr(cfg, "edge_engineering", None)
    enable_pair_recurrence = getattr(ee, "enable_pair_recurrence", False) if ee else False
    enable_fanout_fanin = getattr(ee, "enable_fanout_fanin", False) if ee else False
    enable_op_rarity = getattr(ee, "enable_op_rarity", False) if ee else False
    enable_temporal = getattr(ee, "enable_temporal", False) if ee else False
    
    pair_counts = defaultdict(int)
    src_dests = defaultdict(set)
    dst_sources = defaultdict(set)
    src_op_counts = defaultdict(lambda: defaultdict(int))
    src_counts = defaultdict(int)
    global_op_counts = defaultdict(int)
    last_time_src = {}
    last_time_pair = {}
    ema_10 = defaultdict(float)
    ema_60 = defaultdict(float)
    
    decay_10 = 0.0693  # ln(2) / 10
    decay_60 = 0.0115  # ln(2) / 60
    
    fused_feats = {}
    
    for i, edge in enumerate(raw_edges):
        src = edge["src"]
        dst = edge["dst"]
        op = edge["label"]
        t = edge["time"]
        pair = (src, dst)
        
        feat = []
        
        # 1. Pair recurrence
        if enable_pair_recurrence:
            p_count = pair_counts[pair]
            feat.append(float(p_count))
            feat.append(float(p_count) / (i + 1.0))
            
        # 2. Fanout and Fanin
        if enable_fanout_fanin:
            src_dests[src].add(dst)
            dst_sources[dst].add(src)
            feat.append(float(len(src_dests[src])))
            feat.append(float(len(dst_sources[dst])))
            
        # 3. Op Rarity
        if enable_op_rarity:
            src_op_c = src_op_counts[src][op]
            glob_op_c = global_op_counts[op]
            s_count = src_counts[src]
            
            feat.append(float(src_op_c) / (s_count + 1.0))
            feat.append(float(glob_op_c) / (i + 1.0))
            
        # 4. Temporal / EMAs
        if enable_temporal:
            if src in last_time_src:
                delta_src_sec = (t - last_time_src[src]) / 1e9
                delta_src_sec = max(0.0, delta_src_sec)
                feat.append(math.log1p(delta_src_sec))
                feat.append(1.0)
                
                ema_10[src] = ema_10[src] * math.exp(-decay_10 * delta_src_sec) + 1.0
                ema_60[src] = ema_60[src] * math.exp(-decay_60 * delta_src_sec) + 1.0
            else:
                feat.append(0.0)
                feat.append(0.0)
                ema_10[src] = 1.0
                ema_60[src] = 1.0
                
            if pair in last_time_pair:
                delta_pair_sec = (t - last_time_pair[pair]) / 1e9
                delta_pair_sec = max(0.0, delta_pair_sec)
                feat.append(math.log1p(delta_pair_sec))
                feat.append(1.0)
            else:
                feat.append(0.0)
                feat.append(0.0)
                
            feat.append(ema_10[src])
            feat.append(ema_60[src])
            
        pair_counts[pair] += 1
        src_op_counts[src][op] += 1
        src_counts[src] += 1
        global_op_counts[op] += 1
        last_time_src[src] = t
        last_time_pair[pair] = t
        
        fused_idx = fusion_idxs[i]
        if fused_idx not in fused_feats:
            fused_feats[fused_idx] = feat

    feat_dim = len(next(iter(fused_feats.values()))) if fused_feats else 0
    if feat_dim == 0:
        return None
        
    feats_list = []
    for idx in range(E_fused):
        f_vec = fused_feats.get(idx, [0.0] * feat_dim)
        feats_list.append(f_vec)
        
    return torch.tensor(feats_list, dtype=torch.float)


def feat_inference(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg):
    edge_features = list(map(lambda x: x.strip(), cfg.batching.edge_features.split(",")))
    use_engineered = "engineered" in edge_features

    for path in log_tqdm(sorted_paths, desc="Computing edge embeddings"):
        graph = torch.load(path)
        sorted_edges = graph.edges(data=True, keys=True)

        src, dst, msg, t, y = [], [], [], [], []
        for u, v, k, attr in sorted_edges:
            src.append(int(u))
            dst.append(int(v))
            t.append(int(attr["time"]))
            y.append(int(attr.get("y", 0)))

            if "label" in attr:
                edge_label = etype2oh[attr["label"]]
            else:
                edge_label = torch.zeros_like(etype2oh[list(etype2oh.keys())[0]])

            if indexid2vec is None:
                msg.append(
                    torch.cat(
                        [
                            ntype2oh[graph.nodes[u]["node_type"]],
                            edge_label,
                            ntype2oh[graph.nodes[v]["node_type"]],
                        ]
                    )
                )
            else:
                msg.append(
                    torch.cat(
                        [
                            ntype2oh[graph.nodes[u]["node_type"]],
                            torch.from_numpy(indexid2vec[u]),
                            edge_label,
                            ntype2oh[graph.nodes[v]["node_type"]],
                            torch.from_numpy(indexid2vec[v]),
                        ]
                    )
                )

        engineered_feats = compute_engineered_feats(graph, cfg) if use_engineered else None

        data = CollatableTemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float),
            y=torch.tensor(y).to(torch.long),
            engineered_feats=engineered_feats,
        )

        os.makedirs(out_dir, exist_ok=True)
        file = path.split("/")[-1]
        torch.save(data, os.path.join(out_dir, f"{file}.TemporalData.simple"))


def get_indexid2vec(cfg):
    method = cfg.featurization.used_method.strip()
    if method in ["only_type", "only_ones"]:
        return None
    if method == "alacarte":
        return feat_inference_alacarte.main(cfg)
    if method == "doc2vec":
        return feat_inference_doc2vec.main(cfg)
    if method == "hierarchical_hashing":
        return feat_inference_HFH.main(cfg)
    if method == "word2vec":
        return feat_inference_word2vec.main(cfg)
    if method == "temporal_rw":
        return feat_inference_TRW.main(cfg)
    if method == "flash":
        return feat_inference_flash.main(cfg)
    if method == "fasttext":
        return feat_inference_fasttext.main(cfg)

    raise ValueError(f"Invalid node embedding method {method}")


def main_from_config(cfg):
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map()
    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)

    base_dir = cfg.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)

    # Here we get a mapping {node_id => embedding vector}
    indexid2vec = get_indexid2vec(cfg)

    # Create edges for Train, Val, Test sets
    for split, sorted_paths in split_to_files.items():
        feat_inference(
            indexid2vec=indexid2vec,
            etype2oh=etype2onehot,
            ntype2oh=ntype2onehot,
            sorted_paths=sorted_paths,
            out_dir=os.path.join(cfg.feat_inference._edge_embeds_dir, f"{split}/"),
            cfg=cfg,
        )


def main(cfg):
    multi_dataset_training = cfg.batching.multi_dataset_training
    if not multi_dataset_training:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        trained_model_dir = cfg.featurization._model_dir
        multi_datasets = get_multi_datasets(cfg)
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
            updated_cfg.featurization._model_dir = trained_model_dir

            if should_restart["feat_inference"]:
                main_from_config(updated_cfg)
