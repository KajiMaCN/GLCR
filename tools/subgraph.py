import math
import os
from collections import deque

import numpy as np
import torch


MAX_DIST = 3
NODE_TYPE_COUNT = 3


def _build_neighbors(edge_index, num_nodes):
    neighbors = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighbors[src].add(dst)
    return neighbors


def _build_typed_neighbor_arrays(edge_index, node_type, num_nodes):
    typed = [[set() for _ in range(num_nodes)] for _ in range(NODE_TYPE_COUNT)]
    for src, dst in edge_index.t().tolist():
        dst_type = int(node_type[dst])
        if 0 <= dst_type < NODE_TYPE_COUNT:
            typed[dst_type][src].add(dst)
    typed_arrays = []
    for type_sets in typed:
        typed_arrays.append([np.fromiter(sorted(items), dtype=np.int64) if items else np.empty(0, dtype=np.int64) for items in type_sets])
    return typed_arrays


def _neighbors_without_target(neighbors, node, blocked):
    if blocked in neighbors[node]:
        filtered = set(neighbors[node])
        filtered.discard(blocked)
        return filtered
    return neighbors[node]


def _typed_neighbors(neighbors, node, blocked_peer, node_type, target_type):
    return {
        nei for nei in _neighbors_without_target(neighbors, node, blocked_peer)
        if int(node_type[nei]) == target_type
    }


def _bfs_distances(neighbors, start, blocked_peer, max_hops):
    distances = {start: 0}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for nei in _neighbors_without_target(neighbors, node, blocked_peer if node == start else -1):
            if nei not in distances:
                distances[nei] = dist + 1
                queue.append((nei, dist + 1))
    return distances


def _safe_degree(neighbors, node, blocked_peer):
    return max(1, len(_neighbors_without_target(neighbors, node, blocked_peer)))


def _heuristic_features(src, dst, neighbors, node_type):
    src_neighbors = _neighbors_without_target(neighbors, src, dst)
    dst_neighbors = _neighbors_without_target(neighbors, dst, src)
    common_1hop = src_neighbors & dst_neighbors
    union_1hop = src_neighbors | dst_neighbors

    deg_src = _safe_degree(neighbors, src, dst)
    deg_dst = _safe_degree(neighbors, dst, src)
    aa = 0.0
    ra = 0.0
    cn_type = [0.0] * NODE_TYPE_COUNT
    for node in common_1hop:
        degree = max(2, len(neighbors[node]))
        aa += 1.0 / math.log(float(degree))
        ra += 1.0 / degree
        cn_type[int(node_type[node])] += 1.0

    jaccard = len(common_1hop) / max(1, len(union_1hop))
    pref_attach = float(deg_src * deg_dst)
    return [
        float(deg_src),
        float(deg_dst),
        float(len(common_1hop)),
        float(jaccard),
        float(aa),
        float(ra),
        float(pref_attach),
        *cn_type,
    ]


def _distance_histogram(src, dst, neighbors, node_type, max_hops):
    dist_src = _bfs_distances(neighbors, src, dst, max_hops)
    dist_dst = _bfs_distances(neighbors, dst, src, max_hops)
    sub_nodes = (set(dist_src.keys()) | set(dist_dst.keys())) - {src, dst}

    histogram = torch.zeros((NODE_TYPE_COUNT, MAX_DIST, MAX_DIST), dtype=torch.float32)
    same_type_overlap = torch.zeros(NODE_TYPE_COUNT, dtype=torch.float32)

    for node in sub_nodes:
        node_t = int(node_type[node])
        d1 = min(dist_src.get(node, MAX_DIST), MAX_DIST) - 1
        d2 = min(dist_dst.get(node, MAX_DIST), MAX_DIST) - 1
        d1 = max(0, min(d1, MAX_DIST - 1))
        d2 = max(0, min(d2, MAX_DIST - 1))
        histogram[node_t, d1, d2] += 1.0

        if dist_src.get(node, MAX_DIST + 1) <= max_hops and dist_dst.get(node, MAX_DIST + 1) <= max_hops:
            same_type_overlap[node_t] += 1.0

    return torch.cat([histogram.flatten(), same_type_overlap], dim=0)


def _bridge_context(src, dst, neighbors, node_type, top_k, num_nodes):
    circ_disease = _typed_neighbors(neighbors, src, dst, node_type, target_type=2)
    drug_disease = _typed_neighbors(neighbors, dst, src, node_type, target_type=2)
    shared_disease = list(circ_disease & drug_disease)

    raw_scores = []
    for disease in shared_disease:
        disease_degree = max(2, len(neighbors[disease]))
        src_bridge_degree = max(1, len(circ_disease))
        dst_bridge_degree = max(1, len(drug_disease))
        score = (
            1.0 / math.log1p(float(disease_degree))
            + 0.5 / src_bridge_degree
            + 0.5 / dst_bridge_degree
        )
        raw_scores.append((disease, score))

    raw_scores.sort(key=lambda item: item[1], reverse=True)
    selected = raw_scores[:top_k]

    bridge_ids = torch.full((top_k,), num_nodes, dtype=torch.long)
    bridge_mask = torch.zeros(top_k, dtype=torch.float32)
    bridge_prior = torch.zeros(top_k, dtype=torch.float32)
    bridge_feat = torch.zeros((top_k, 3), dtype=torch.float32)
    if selected:
        scores = torch.tensor([score for _, score in selected], dtype=torch.float32)
        scores = scores / scores.sum().clamp_min(1e-8)
        for idx, (disease, _) in enumerate(selected):
            disease_degree = max(2, len(neighbors[disease]))
            bridge_ids[idx] = disease
            bridge_mask[idx] = 1.0
            bridge_prior[idx] = scores[idx]
            bridge_feat[idx] = torch.tensor(
                [
                    float(1.0 / math.log1p(float(disease_degree))),
                    float(1.0 / max(1, len(circ_disease))),
                    float(1.0 / max(1, len(drug_disease))),
                ],
                dtype=torch.float32,
            )

    bridge_stats = torch.tensor(
        [
            float(len(shared_disease)),
            float(len(circ_disease)),
            float(len(drug_disease)),
            float(len(shared_disease) / max(1, len(circ_disease | drug_disease))),
        ],
        dtype=torch.float32,
    )
    return bridge_ids, bridge_mask, bridge_prior, bridge_feat, bridge_stats


def compute_link_context(data, links, num_hops=2, top_k=8):
    if isinstance(links, torch.Tensor):
        links = links.cpu()
    links = torch.as_tensor(links, dtype=torch.long)

    num_nodes = int(data.num_nodes)
    node_type = data.node_type.cpu()
    edge_index = data.edge_index.cpu()

    # Fast path for tripartite graphs: use typed-neighbor precomputation instead of per-link BFS.
    typed_neighbors = _build_typed_neighbor_arrays(edge_index, node_type, num_nodes)
    total_degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float32).cpu().numpy()
    mediator_degree = np.maximum(2.0, total_degree[np.where(node_type.numpy() == 2)[0]] if torch.any(node_type == 2) else np.array([], dtype=np.float32))
    mediator_degree_map = np.ones(num_nodes, dtype=np.float32) * 2.0
    mediator_nodes = np.where(node_type.numpy() == 2)[0]
    if len(mediator_nodes) > 0:
        mediator_degree_map[mediator_nodes] = np.maximum(2.0, total_degree[mediator_nodes])
    inv_log_degree = 1.0 / np.log1p(mediator_degree_map)
    inv_degree = 1.0 / mediator_degree_map

    num_links = links.size(0)
    features = torch.zeros((num_links, 44), dtype=torch.float32)
    bridge_ids = torch.full((num_links, top_k), num_nodes, dtype=torch.long)
    bridge_mask = torch.zeros((num_links, top_k), dtype=torch.float32)
    bridge_prior = torch.zeros((num_links, top_k), dtype=torch.float32)
    bridge_feat = torch.zeros((num_links, top_k, 3), dtype=torch.float32)
    bridge_stats = torch.zeros((num_links, 4), dtype=torch.float32)

    mediator_neighbors = typed_neighbors[2]
    for idx, (src, dst) in enumerate(links.tolist()):
        src_mediator = mediator_neighbors[src]
        dst_mediator = mediator_neighbors[dst]
        shared = np.intersect1d(src_mediator, dst_mediator, assume_unique=True)
        shared_count = int(shared.size)
        union_count = int(src_mediator.size + dst_mediator.size - shared_count)
        deg_src = max(1.0, float(total_degree[src]))
        deg_dst = max(1.0, float(total_degree[dst]))

        aa = float(inv_log_degree[shared].sum()) if shared_count > 0 else 0.0
        ra = float(inv_degree[shared].sum()) if shared_count > 0 else 0.0
        jaccard = float(shared_count / max(1, union_count))
        pref_attach = float(deg_src * deg_dst)

        stats = np.array(
            [
                float(shared_count),
                float(src_mediator.size),
                float(dst_mediator.size),
                float(shared_count / max(1, union_count)),
            ],
            dtype=np.float32,
        )
        bridge_stats[idx] = torch.from_numpy(stats)

        histogram = np.zeros((NODE_TYPE_COUNT, MAX_DIST, MAX_DIST), dtype=np.float32)
        same_type_overlap = np.zeros(NODE_TYPE_COUNT, dtype=np.float32)
        histogram[2, 0, 0] = float(shared_count)
        histogram[2, 0, 2] = float(max(0, src_mediator.size - shared_count))
        histogram[2, 2, 0] = float(max(0, dst_mediator.size - shared_count))
        same_type_overlap[2] = float(shared_count)

        heuristic = np.array(
            [
                deg_src,
                deg_dst,
                float(shared_count),
                jaccard,
                aa,
                ra,
                pref_attach,
                0.0,
                0.0,
                float(shared_count),
            ],
            dtype=np.float32,
        )
        feature_vec = np.concatenate([heuristic, histogram.reshape(-1), same_type_overlap, stats], axis=0)
        features[idx] = torch.from_numpy(feature_vec)

        if shared_count > 0:
            score = inv_log_degree[shared] + (0.5 / max(1, src_mediator.size)) + (0.5 / max(1, dst_mediator.size))
            order = np.argsort(-score)[:top_k]
            selected = shared[order]
            selected_scores = score[order].astype(np.float32)
            selected_scores /= np.maximum(selected_scores.sum(), 1e-8)
            count = selected.shape[0]
            bridge_ids[idx, :count] = torch.from_numpy(selected)
            bridge_mask[idx, :count] = 1.0
            bridge_prior[idx, :count] = torch.from_numpy(selected_scores)
            bridge_feat[idx, :count, 0] = torch.from_numpy(inv_log_degree[selected].astype(np.float32))
            bridge_feat[idx, :count, 1] = float(1.0 / max(1, src_mediator.size))
            bridge_feat[idx, :count, 2] = float(1.0 / max(1, dst_mediator.size))

    return {
        'features': features,
        'bridge_ids': bridge_ids,
        'bridge_mask': bridge_mask,
        'bridge_prior': bridge_prior,
        'bridge_feat': bridge_feat,
        'bridge_stats': bridge_stats,
    }


def _context_payload_matches(payload, num_links, top_k):
    required_keys = {
        'features',
        'bridge_ids',
        'bridge_mask',
        'bridge_prior',
        'bridge_feat',
        'bridge_stats',
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        return False

    for key in required_keys:
        if not torch.is_tensor(payload[key]):
            return False

    if payload['features'].dim() != 2 or payload['features'].size(0) != num_links:
        return False
    if tuple(payload['bridge_ids'].shape) != (num_links, top_k):
        return False
    if tuple(payload['bridge_mask'].shape) != (num_links, top_k):
        return False
    if tuple(payload['bridge_prior'].shape) != (num_links, top_k):
        return False
    if tuple(payload['bridge_feat'].shape) != (num_links, top_k, 3):
        return False
    if tuple(payload['bridge_stats'].shape) != (num_links, 4):
        return False
    return True


def load_or_compute_link_context(data, links, cache_path, num_hops=2, top_k=8):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    num_links = int(torch.as_tensor(links).shape[0])
    if os.path.exists(cache_path):
        try:
            payload = torch.load(cache_path, map_location='cpu')
        except Exception:
            payload = None
        if _context_payload_matches(payload, num_links, top_k):
            return payload

    payload = compute_link_context(data, links, num_hops=num_hops, top_k=top_k)
    torch.save(payload, cache_path)
    return payload
