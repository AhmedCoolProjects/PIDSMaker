import json
import os
from datetime import datetime, timedelta

import networkx as nx
import torch

from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    init_database_connection,
    log,
    log_start,
    ns_time_to_datetime_US,
    stream_query,
)


def get_node_list(cur, connect, cfg):
    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)

    uuid2idx = {}
    uuid2type = {}
    uuid2name = {}
    hash2uuid = {}

    # netflow — node_uuid | hash_id | src_addr | src_port | dst_addr | dst_port | index_id
    # Stream rows via a server-side cursor instead of the previous
    # execute()+fetchmany() loop. The old form buffered the whole result set
    # client-side; on E5-scale node tables that alone can be >50GB.
    for i in stream_query(connect, "select * from netflow_node_table;", name="pids_magic_netflow"):
        attrs = {
            "type": "netflow",
            "local_ip": str(i[2]),
            "local_port": str(i[3]),
            "remote_ip": str(i[4]),
            "remote_port": str(i[5]),
        }
        node_uuid = str(i[0])
        hash_id = str(i[1])
        index_id = int(i[-1])

        features_used = []
        for label_used in node_label_features["netflow"]:
            features_used.append(attrs[label_used])
        label_str = " ".join(features_used)

        uuid2idx[node_uuid] = index_id
        uuid2type[node_uuid] = attrs["type"]
        uuid2name[node_uuid] = label_str
        hash2uuid[hash_id] = node_uuid

    # subject — node_uuid | hash_id | path | cmd | index_id
    for i in stream_query(connect, "select * from subject_node_table;", name="pids_magic_subject"):
        attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
        node_uuid = str(i[0])
        hash_id = str(i[1])
        index_id = int(i[-1])
        features_used = []
        for label_used in node_label_features["subject"]:
            features_used.append(attrs[label_used])
        label_str = " ".join(features_used)

        uuid2idx[node_uuid] = index_id
        uuid2type[node_uuid] = attrs["type"]
        uuid2name[node_uuid] = label_str
        hash2uuid[hash_id] = node_uuid

    # file — node_uuid | hash_id | path | index_id
    for i in stream_query(connect, "select * from file_node_table;", name="pids_magic_file"):
        attrs = {"type": "file", "path": str(i[2])}
        node_uuid = str(i[0])
        hash_id = str(i[1])
        index_id = int(i[-1])
        features_used = []
        for label_used in node_label_features["file"]:
            features_used.append(attrs[label_used])
        label_str = " ".join(features_used)

        uuid2idx[node_uuid] = index_id
        uuid2type[node_uuid] = attrs["type"]
        uuid2name[node_uuid] = label_str
        hash2uuid[hash_id] = node_uuid

    return uuid2idx, uuid2type, uuid2name, hash2uuid


def generate_graphs(cur, connect, uuid2type, graph_out_dir, hash2uuid, cfg):
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map()
    include_edge_type = rel2id
    node_type_dict = ntype2id

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]

    # In test mode, we ensure to get 1 TW in each set
    dates = get_dates_from_cfg(cfg)

    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start)
            end_ns_timestamp = datetime_to_ns_time_US(stop)
            sql = (
                "select * from event_table "
                f"where timestamp_rec>'{start_ns_timestamp}' and timestamp_rec<'{end_ns_timestamp}' "
                "ORDER BY timestamp_rec;"
            )

            # Stream rows from a server-side cursor so the entire day of events
            # is not buffered client-side before filtering.
            events_list = [
                row
                for row in stream_query(connect, sql, name="pids_magic_events")
                if row[2] in include_edge_type
            ]

            if not events_list:
                continue

            start_time = events_list[0][-2]
            temp_list = []
            BATCH = 1024
            window_size_in_sec = cfg.construction.time_window_size * 60_000_000_000

            last_batch = False
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
                    last_batch = True

                if (batch_edges[-1][-2] > start_time + window_size_in_sec) or last_batch:
                    time_interval = (
                        ns_time_to_datetime_US(start_time)
                        + "~"
                        + ns_time_to_datetime_US(batch_edges[-1][-2])
                    )

                    log(f"Start create edge fused time window graph for {time_interval}")

                    g = nx.DiGraph()
                    node_visited = set()

                    for event_tuple in events_list:
                        (
                            src_node,
                            src_index_id,
                            operation,
                            dst_node,
                            dst_index_id,
                            event_uuid,
                            timestamp_rec,
                            _id,
                        ) = event_tuple
                        if src_index_id not in node_visited:
                            g.add_node(
                                int(src_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[src_node]]],
                            )
                            node_visited.add(src_index_id)
                        if dst_index_id not in node_visited:
                            g.add_node(
                                int(dst_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[dst_node]]],
                            )
                            node_visited.add(dst_index_id)
                        if not g.has_edge(int(src_index_id), int(dst_index_id)):
                            g.add_edge(
                                int(src_index_id),
                                int(dst_index_id),
                                type=include_edge_type[operation],
                            )

                    date_dir = f"{graph_out_dir}/graph_{date}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    print(f"Saving graph for {time_interval}")
                    torch.save(g, graph_name)

                    start_time = batch_edges[-1][-2]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break
    return


def main(cfg):
    log_start(__file__)
    cur, connect = init_database_connection(cfg)
    uuid2idx, uuid2type, uuid2name, hash2uuid = get_node_list(cur=cur, connect=connect, cfg=cfg)

    os.makedirs(cfg.construction._magic_dir, exist_ok=True)
    os.makedirs(cfg.construction._magic_graphs_dir, exist_ok=True)
    file_out_dir = cfg.construction._magic_dir
    graph_out_dir = cfg.construction._magic_graphs_dir

    with open(os.path.join(file_out_dir, "names.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2name, fw)
    with open(os.path.join(file_out_dir, "types.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2type, fw)

    generate_graphs(
        cur=cur,
        connect=connect,
        uuid2type=uuid2type,
        graph_out_dir=graph_out_dir,
        hash2uuid=hash2uuid,
        cfg=cfg,
    )

    del uuid2idx, uuid2type, uuid2name, hash2uuid
