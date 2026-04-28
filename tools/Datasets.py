import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tools.utils import get_abspath


DATASET_CONFIGS = {
    "ncRNADrug": {
        "node_types": ("ncrna", "drug", "phenotype"),
        "primary_edge": ("edges_ncrna_drug.csv", "ncrnaId", "drugId"),
        "support_edges": [
            ("edges_ncrna_phenotype.csv", "ncrnaId", "phenotypeId"),
            ("edges_drug_phenotype.csv", "drugId", "phenotypeId"),
        ],
    },
    "openTargets": {
        "node_types": ("drug", "target", "disease"),
        "primary_edge": ("edges_drug_target.csv", "drugId", "targetId"),
        "support_edges": [
            ("edges_drug_disease.csv", "drugId", "diseaseId"),
            ("edges_target_disease.csv", "targetId", "diseaseId"),
        ],
    },
    "CTD": {
        "node_types": ("chemical", "gene", "disease"),
        "primary_edge": ("edges_chemical_gene.csv", "chemicalId", "geneId"),
        "support_edges": [
            ("edges_chemical_disease.csv", "chemicalId", "diseaseId"),
            ("edges_gene_disease.csv", "geneId", "diseaseId"),
        ],
    },
    "PrimeKG": {
        "node_types": ("drug", "target", "disease"),
        "primary_edge": ("edges_drug_target.csv", "drugId", "targetId"),
        "support_edges": [
            ("edges_drug_disease.csv", "drugId", "diseaseId"),
            ("edges_target_disease.csv", "targetId", "diseaseId"),
        ],
    },
}


class Datasets:
    def __init__(self, folder):
        self.folder = folder
        self.abs_folder = get_abspath(folder)

    def remove_test_edges(self, data, test_pos, train_links=None, train_labels=None, blocked_nodes=None):
        data_copy = data.clone()
        test_pos_edges = {tuple(edge) for edge in np.asarray(test_pos, dtype=np.int64).tolist()}
        reverse_test_pos_edges = {(dst, src) for src, dst in test_pos_edges}

        blocked_node_set = set()
        if blocked_nodes is not None:
            blocked_node_set = {int(node_id) for node_id in np.asarray(blocked_nodes, dtype=np.int64).reshape(-1).tolist()}

        all_edges = data_copy.edge_index.t().cpu().numpy().tolist()
        edge_mask = np.array(
            [tuple(edge) not in test_pos_edges and tuple(edge) not in reverse_test_pos_edges for edge in all_edges],
            dtype=bool,
        )
        if blocked_node_set:
            blocked_mask = np.array(
                [
                    int(edge[0]) not in blocked_node_set and int(edge[1]) not in blocked_node_set
                    for edge in all_edges
                ],
                dtype=bool,
            )
            edge_mask &= blocked_mask
        data_copy.edge_index = data_copy.edge_index[:, edge_mask]

        if train_links is None or train_labels is None:
            return data_copy

        train_mask = np.array(
            [tuple(edge) not in test_pos_edges for edge in np.asarray(train_links, dtype=np.int64).tolist()],
            dtype=bool,
        )
        if blocked_node_set:
            blocked_train_mask = np.array(
                [
                    int(edge[0]) not in blocked_node_set and int(edge[1]) not in blocked_node_set
                    for edge in np.asarray(train_links, dtype=np.int64).tolist()
                ],
                dtype=bool,
            )
            train_mask &= blocked_train_mask
        return data_copy, train_links[train_mask, :], train_labels[train_mask]

    def process(self, dataset):
        if dataset not in DATASET_CONFIGS:
            supported = ", ".join(sorted(DATASET_CONFIGS))
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")
        return self._process_tripartite(dataset, DATASET_CONFIGS[dataset])

    def _process_tripartite(self, dataset, cfg):
        data_folder = os.path.join(self.abs_folder, dataset)
        feature_path = os.path.join(data_folder, "all_node_features.csv")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        feature_df = pd.read_csv(feature_path, low_memory=False).sort_values("global_idx").reset_index(drop=True)
        feature_cols = [col for col in feature_df.columns if col.startswith("f_")]
        if not feature_cols:
            raise ValueError(f"No feature columns found in {feature_path}")

        node_features = feature_df[feature_cols].to_numpy(dtype=np.float32)
        node_types = cfg["node_types"]
        node_type_map = {name: idx for idx, name in enumerate(node_types)}
        node_type = feature_df["node_type"].map(node_type_map).to_numpy(dtype=np.int64)
        idx_map = dict(zip(feature_df["node_id"].astype(str), feature_df["global_idx"]))

        def mapped_edges(filename, src_col, dst_col):
            frame = pd.read_csv(os.path.join(data_folder, filename))
            frame[src_col] = frame[src_col].astype(str)
            frame[dst_col] = frame[dst_col].astype(str)
            mapped = pd.DataFrame(
                {
                    "src": frame[src_col].map(idx_map),
                    "dst": frame[dst_col].map(idx_map),
                }
            ).dropna().drop_duplicates()
            return mapped[["src", "dst"]].to_numpy(dtype=np.int64)

        primary_file, primary_src_col, primary_dst_col = cfg["primary_edge"]
        pos_index = mapped_edges(primary_file, primary_src_col, primary_dst_col)
        support_edges = [
            mapped_edges(filename, src_col, dst_col)
            for filename, src_col, dst_col in cfg["support_edges"]
        ]

        src_nodes = feature_df.loc[feature_df["node_type"] == node_types[0], "global_idx"].to_numpy(dtype=np.int64)
        dst_nodes = feature_df.loc[feature_df["node_type"] == node_types[1], "global_idx"].to_numpy(dtype=np.int64)
        all_pairs = np.stack(np.meshgrid(src_nodes, dst_nodes, indexing="ij"), axis=-1).reshape(-1, 2)
        pos_set = {tuple(edge) for edge in pos_index.tolist()}
        neg_mask = np.array([tuple(edge) not in pos_set for edge in all_pairs.tolist()], dtype=bool)
        neg_index = all_pairs[neg_mask]

        x = torch.tensor(node_features, dtype=torch.float32)
        graph_edges = np.vstack([pos_index, *support_edges])
        edge_index = torch.tensor(graph_edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.shape[0])
        data = Data(x=x, edge_index=edge_index, node_type=torch.tensor(node_type, dtype=torch.long))

        return data, pos_index.astype(np.int64), neg_index.astype(np.int64)
