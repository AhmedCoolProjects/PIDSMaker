"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation.
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    get_split_to_files,
    init_database_connection,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    stream_query,
    stringtomd5,
)


def compute_indexid2msg(cfg):
    """Compute mapping from node index IDs to node types and feature labels.

    Queries PostgreSQL database for all nodes (netflow, subject/process, file) and
    extracts their attributes to create feature labels based on configuration.

    Args:
        cfg: Configuration with database connection and feature settings

    Returns:
        dict: Mapping {index_id: [node_type, label_string]} where:
            - index_id: Database node identifier
            - node_type: One of 'netflow', 'subject', 'file'
            - label_string: Feature label (hashed or plaintext depending on config)
    """
    cur, connect = init_database_connection(cfg)

    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    indexid2msg = {}

    def get_label_str_from_features(attrs, node_type):
        """Extract feature label from node attributes based on configured features.

        Args:
            attrs: Dictionary of node attributes
            node_type: Type of node ('netflow', 'subject', 'file')

        Returns:
            str: Space-separated feature string, optionally hashed
        """
        label_str = " ".join([attrs[label_used] for label_used in node_label_features[node_type]])
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    # netflow — stream rows so we never hold the whole table in Python memory
    n_netflow = 0
    for i in stream_query(connect, "select * from netflow_node_table;", name="pids_netflow"):
        attrs = {
            "type": "netflow",
            "local_ip": str(i[2]),
            "local_port": str(i[3]),
            "remote_ip": str(i[4]),
            "remote_port": str(i[5]),
        }
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        n_netflow += 1
    log(f"Number of netflow nodes: {n_netflow}")

    # subject
    n_subject = 0
    for i in stream_query(connect, "select * from subject_node_table;", name="pids_subject"):
        attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        n_subject += 1
    log(f"Number of process nodes: {n_subject}")

    # file
    n_file = 0
    for i in stream_query(connect, "select * from file_node_table;", name="pids_file"):
        attrs = {"type": "file", "path": str(i[2])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        n_file += 1
    log(f"Number of file nodes: {n_file}")

    return indexid2msg  # {index_id: [node_type, msg]}


def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """Save filtered node index-to-feature mapping to disk.

    Filters out nodes not used in any train/val/test graphs (due to excluded edge types)
    before saving to avoid downstream errors during featurization.

    Note: Must be called after graph construction to ensure only used nodes are saved.

    Args:
        indexid2msg: Full node mapping from compute_indexid2msg()
        split2nodes: Mapping of splits to their node sets
        cfg: Configuration with output directory path
    """
    all_nodes = set().union(*(split2nodes[split] for split in ["train", "val", "test"]))
    indexid2msg = {k: v for k, v in indexid2msg.items() if k in all_nodes}

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg):
    """Compute and save mapping of dataset splits to their node sets.

    Loads all graphs from train/val/test splits and collects unique node IDs
    appearing in each split. Used to filter node features and track split membership.

    Args:
        cfg: Configuration with graph directory and split file paths

    Returns:
        dict: Mapping of split names to node sets:
            {'train': {node_ids}, 'val': {node_ids}, 'test': {node_ids}}
    """
    split_to_files = get_split_to_files(cfg, cfg.construction._graphs_dir)
    split2nodes = defaultdict(set)

    # Stream one graph at a time. The previous list-comprehension form held
    # every NetworkX MultiDiGraph for a split in RAM simultaneously, which
    # is multi-GB-per-split on large datasets and produced spikes during a
    # step that only needs the node id sets.
    for split, files in split_to_files.items():
        for path in log_tqdm(files, desc=f"Check nodes in {split} set"):
            G = torch.load(path)
            split2nodes[split].update(G.nodes())
            del G
    split2nodes = dict(split2nodes)

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


def gen_edge_fused_tw(indexid2msg, cfg):
    """Generate time-windowed provenance graphs from database events.

    Main graph construction function that:
    1. Queries database for events in time windows
    2. Optionally fuses consecutive edges of same type between node pairs
    3. Optionally adds attack mimicry events for data augmentation
    4. Builds NetworkX MultiDiGraphs with node attributes and edge metadata
    5. Saves graphs to disk organized by day and time window

    Args:
        indexid2msg: Node index to [type, label] mapping from compute_indexid2msg()
        cfg: Configuration with:
            - Database connection settings
            - Time window parameters (size, dates)
            - Edge type filtering (rel2id)
            - Mimicry settings (mimicry_edge_num)
            - Output directory paths
    """
    cur, connect = init_database_connection(cfg)
    rel2id = get_rel2id(cfg)
    include_edge_type = rel2id

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        attack_mimicry_events = mimicry.gen_mimicry_edges(cfg)
    else:
        attack_mimicry_events = defaultdict(list)

    def _build_and_save_graph(temp_list, start_time, end_event_ts, date, cfg, indexid2msg):
        """Build a NetworkX MultiDiGraph from the buffered window events and save it.

        Extracted as a helper so the streaming driver below stays readable. Logic
        mirrors the original inline build: optional edge fusion, then graph
        materialization, then torch.save.
        """
        time_interval = (
            ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(end_event_ts)
        )

        node_info = {}
        edge_list = []
        if cfg.construction.fuse_edge:
            edge_info = {}
            for (
                _src_node,
                src_index_id,
                operation,
                _dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in temp_list:
                if src_index_id not in node_info:
                    node_type, label = indexid2msg[src_index_id]
                    node_info[src_index_id] = {"label": label, "node_type": node_type}
                if dst_index_id not in node_info:
                    node_type, label = indexid2msg[dst_index_id]
                    node_info[dst_index_id] = {"label": label, "node_type": node_type}

                if (src_index_id, dst_index_id) not in edge_info:
                    edge_info[(src_index_id, dst_index_id)] = []

                edge_info[(src_index_id, dst_index_id)].append(
                    (timestamp_rec, operation, event_uuid)
                )

            for (src, dst), data in edge_info.items():
                sorted_data = sorted(data, key=lambda x: x[0])
                operation_list = [entry[1] for entry in sorted_data]

                indices = []
                current_type = None
                current_start_index = None

                for idx, item in enumerate(operation_list):
                    if item == current_type:
                        continue
                    else:
                        if current_type is not None and current_start_index is not None:
                            indices.append(current_start_index)
                        current_type = item
                        current_start_index = idx

                if current_type is not None and current_start_index is not None:
                    indices.append(current_start_index)

                for k in indices:
                    edge_list.append(
                        {
                            "src": src,
                            "dst": dst,
                            "time": sorted_data[k][0],
                            "label": sorted_data[k][1],
                            "event_uuid": sorted_data[k][2],
                        }
                    )
        else:
            for (
                _src_node,
                src_index_id,
                operation,
                _dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in temp_list:
                if src_index_id not in node_info:
                    node_type, label = indexid2msg[src_index_id]
                    node_info[src_index_id] = {"label": label, "node_type": node_type}
                if dst_index_id not in node_info:
                    node_type, label = indexid2msg[dst_index_id]
                    node_info[dst_index_id] = {"label": label, "node_type": node_type}

                edge_list.append(
                    {
                        "src": src_index_id,
                        "dst": dst_index_id,
                        "time": timestamp_rec,
                        "label": operation,
                        "event_uuid": event_uuid,
                    }
                )

        graph = nx.MultiDiGraph()

        for node, info in node_info.items():
            graph.add_node(node, node_type=info["node_type"], label=info["label"])

        for i, edge in enumerate(edge_list):
            graph.add_edge(
                edge["src"],
                edge["dst"],
                event_uuid=edge["event_uuid"],
                time=edge["time"],
                label=edge["label"],
                y=0,
            )

            # For unit tests, we only want few edges
            NUM_TEST_EDGES = 2000
            if cfg._test_mode and i >= NUM_TEST_EDGES:
                break

        date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
        os.makedirs(date_dir, exist_ok=True)
        graph_name = f"{date_dir}/{time_interval}"

        torch.save(graph, graph_name)

    # In test mode, we ensure to get 1 TW in each set
    dates = get_dates_from_cfg(cfg)

    log("Building graphs...")
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

            attack_index = 0
            mimicry_events = []
            for attack_tuple in cfg.dataset.attack_to_time_window:
                attack = attack_tuple[0]
                attack_start_time = datetime_to_ns_time_US(attack_tuple[1])
                attack_end_time = datetime_to_ns_time_US(attack_tuple[2])

                if mimicry_edge_num > 0 and (
                    attack_start_time >= start_ns_timestamp and attack_end_time <= end_ns_timestamp
                ):
                    log(
                        f"Insert mimicry events into attack {attack_index} when building graphs from {date_start} to {date_stop}"
                    )
                    mimicry_events.extend(attack_mimicry_events[attack_index])
                attack_index += 1

            sql = (
                "select * from event_table "
                f"where timestamp_rec>'{start_ns_timestamp}' and timestamp_rec<'{end_ns_timestamp}' "
                "ORDER BY timestamp_rec, event_uuid;"
            )

            # Stream events from Postgres instead of fetchall — for huge days
            # (TRACE_E5, FIVEDIRECTIONS_E5, ...) the previous fetchall held
            # tens of millions of tuples in RAM and was duplicated into
            # `events_list`. We now consume the cursor row-by-row, filter on
            # the fly, append mimicry events at the end, and emit graphs once
            # the rolling buffer crosses the time-window threshold.
            def filtered_event_stream(_sql=sql, _mimicry=mimicry_events):
                for row in stream_query(connect, _sql, name="pids_events"):
                    if row[2] in include_edge_type:  # operation filter
                        yield row
                for row in _mimicry:
                    if row[2] in include_edge_type:
                        yield row

            BATCH = 1024
            window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

            stream = filtered_event_stream()
            temp_list = []
            start_time = None  # first event ts of the current window
            stream_exhausted = False

            while not stream_exhausted:
                # Pull up to BATCH events; stop early if stream ends.
                batch_buf = []
                for _ in range(BATCH):
                    try:
                        batch_buf.append(next(stream))
                    except StopIteration:
                        stream_exhausted = True
                        break

                if not batch_buf:
                    break  # no rows at all (or no more), nothing to flush
                if start_time is None:
                    start_time = batch_buf[0][-2]

                temp_list.extend(batch_buf)
                last_event_ts = batch_buf[-1][-2]
                last_batch = stream_exhausted

                if (last_event_ts > start_time + window_size_in_ns) or last_batch:
                    _build_and_save_graph(
                        temp_list=temp_list,
                        start_time=start_time,
                        end_event_ts=last_event_ts,
                        date=date,
                        cfg=cfg,
                        indexid2msg=indexid2msg,
                    )

                    start_time = last_event_ts
                    temp_list.clear()

                    # For unit tests, we only keep edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break


def main(cfg):
    """Main construction pipeline: build graphs from database and save metadata.

    Execution flow:
    1. Extract node features from database (compute_indexid2msg)
    2. Build time-windowed graphs from events (gen_edge_fused_tw)
    3. Compute dataset split node memberships (compute_and_save_split2nodes)
    4. Save filtered node features (save_indexid2msg)

    Args:
        cfg: Configuration object with all construction parameters
    """
    log_start(__file__)

    indexid2msg = compute_indexid2msg(cfg=cfg)

    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg)

    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)
