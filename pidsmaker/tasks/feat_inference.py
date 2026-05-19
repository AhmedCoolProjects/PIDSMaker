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
import math
from collections import defaultdict
from pidsmaker.utils.data_utils import CollatableTemporalData
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


def feat_inference(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg):
    for path in log_tqdm(sorted_paths, desc="Computing edge embeddings"):
        graph = torch.load(path)
        sorted_edges = graph.edges(data=True, keys=True)
        
        ee_enabled = getattr(cfg, "edge_engineering", False) and getattr(cfg.edge_engineering, "enabled", False)
        if ee_enabled:
            ema_alpha = cfg.edge_engineering.ema_alpha
            node_dst_fanin = defaultdict(set)
            node_src_fanout = defaultdict(set)
            pair_recurrence = defaultdict(int)
            src_op_counts = defaultdict(lambda: defaultdict(int))
            src_last_time = defaultdict(lambda: None)
            pair_last_time = defaultdict(lambda: None)
            src_burst_ema = defaultdict(float)
            eng_feats = []

        src, dst, msg, t, y = [], [], [], [], []
        for u, v, k, attr in sorted_edges:
            src.append(int(u))
            dst.append(int(v))
            t.append(int(attr["time"]))
            y.append(int(attr.get("y", 0)))

            # If the graph structure has been changed in transformation, we may loose
            # the edge label
            if "label" in attr:
                edge_label = etype2oh[attr["label"]]
            else:
                edge_label = torch.zeros_like(etype2oh[list(etype2oh.keys())[0]])

            # Only types
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

            # Types + node embeddings
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

            if ee_enabled:
                pair = (int(u), int(v))
                pair_rec = pair_recurrence[pair]
                pair_recurrence[pair] += 1
                
                src_fanout = len(node_src_fanout[int(u)])
                dst_fanin = len(node_dst_fanin[int(v)])
                node_src_fanout[int(u)].add(int(v))
                node_dst_fanin[int(v)].add(int(u))
                
                op = attr.get("label", None)
                total_ops = sum(src_op_counts[int(u)].values())
                op_freq = src_op_counts[int(u)][op] if op is not None else 0
                op_rarity = op_freq / total_ops if total_ops > 0 else 0.0
                
                entropy = 0.0
                if total_ops > 0:
                    for count in src_op_counts[int(u)].values():
                        if count > 0:
                            p = count / total_ops
                            entropy -= p * math.log2(p)
                        
                if op is not None:
                    src_op_counts[int(u)][op] += 1
                    
                curr_t = int(attr["time"])
                dt_src = curr_t - src_last_time[int(u)] if src_last_time[int(u)] is not None else 0
                dt_pair = curr_t - pair_last_time[pair] if pair_last_time[pair] is not None else 0
                
                src_last_time[int(u)] = curr_t
                pair_last_time[pair] = curr_t
                
                if dt_src > 0:
                    src_burst_ema[int(u)] = ema_alpha * dt_src + (1 - ema_alpha) * src_burst_ema[int(u)]
                ema_val = src_burst_ema[int(u)]
                
                feats = [pair_rec, src_fanout, dst_fanin, op_rarity, entropy, dt_src, dt_pair, ema_val]
                
                if getattr(cfg.edge_engineering, "normalize_counts", False):
                    feats[0] = math.log1p(feats[0])
                    feats[1] = math.log1p(feats[1])
                    feats[2] = math.log1p(feats[2])
                    
                eng_feats.append(feats)

        if ee_enabled:
            eng_tensor = torch.tensor(eng_feats, dtype=torch.float)
            if getattr(cfg.edge_engineering, "standardize_time", False) and eng_tensor.shape[0] > 0:
                max_t = eng_tensor[:, 5:8].max(dim=0, keepdim=True)[0] + 1e-6
                eng_tensor[:, 5:8] /= max_t

        data = CollatableTemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float),
            y=torch.tensor(y).to(torch.long),
        )
        if ee_enabled and eng_tensor.shape[0] > 0:
            data.engineered_feats = eng_tensor

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
