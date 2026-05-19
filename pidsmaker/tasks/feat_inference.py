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
from pidsmaker.tasks.engineered_feats import EngineeredFeatureComputer
from pidsmaker.utils.data_utils import CollatableTemporalData
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


def feat_inference(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg, rel2id=None):
    edge_engineering_enabled = getattr(cfg.batching, "edge_engineering", None) and cfg.batching.edge_engineering.enabled

    for path in log_tqdm(sorted_paths, desc="Computing edge embeddings"):
        graph = torch.load(path)
        sorted_edges = graph.edges(data=True, keys=True)

        src, dst, msg, t, y = [], [], [], [], []
        engineered_feats_list = []
        computer = None
        op2id = None

        if edge_engineering_enabled:
            eng_cfg = cfg.batching.edge_engineering
            num_op_types = len(rel2id) if rel2id is not None else 0
            ema_alpha = eng_cfg.ema_alpha
            use_log1p = eng_cfg.normalization.strip() == "log1p"
            families = eng_cfg.feature_families
            enabled = set()
            for name, enabled_flag in [
                ("pair_recurrence", families.pair_recurrence),
                ("source_fanout", families.source_fanout),
                ("dst_fanin", families.dst_fanin),
                ("op_rarity", families.op_rarity),
                ("temporal", families.temporal),
                ("burstiness", families.burstiness),
                ("entropy", families.entropy),
            ]:
                if enabled_flag:
                    enabled.add(name)
            computer = EngineeredFeatureComputer(num_op_types, ema_alpha, use_log1p, enabled)
            op2id = {
                op: idx for op, idx in rel2id.items()
                if isinstance(op, str) and isinstance(idx, int)
            }
            if op2id:
                min_id = min(op2id.values())
                op2id = {op: idx - min_id for op, idx in op2id.items()}

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

            if computer is not None and op2id is not None:
                op_type_idx = op2id.get(attr.get("label"), 0)
                eng_feat = computer.step(
                    src=int(u),
                    dst=int(v),
                    op_type=op_type_idx,
                    timestamp_ns=int(attr["time"]),
                )
                engineered_feats_list.append(eng_feat)

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

        extra_kwargs = {}
        if engineered_feats_list:
            extra_kwargs["engineered_feats"] = (
                torch.stack(engineered_feats_list).to(torch.float)
            )

        data = CollatableTemporalData(
            src=torch.tensor(src).to(torch.long),
            dst=torch.tensor(dst).to(torch.long),
            t=torch.tensor(t).to(torch.long),
            msg=torch.vstack(msg).to(torch.float),
            y=torch.tensor(y).to(torch.long),
            **extra_kwargs,
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
            rel2id=rel2id,
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
