import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from model.GLCR import GLCRClassifier, GLCRModel
from paper_configs import PAPER_DATASETS, RELEASE_LAMBDA_BR, RELEASE_LAMBDA_PERT, apply_release_defaults
from tools.Datasets import Datasets
from tools.subgraph import load_or_compute_link_context


def setup_logger(log_dir='logs', log_tag=''):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    safe_tag = str(log_tag).strip().replace(' ', '_') if log_tag else ''
    filename = f'run_{timestamp}.log' if not safe_tag else f'run_{timestamp}_{safe_tag}.log'
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger('GLCR')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('Log file: %s', log_path)
    return logger, log_path


def get_args():
    ap = argparse.ArgumentParser(description='GLCR paper-release training script')
    ap.add_argument('--dataset', choices=PAPER_DATASETS, default=PAPER_DATASETS[0], help='paper dataset key under datasets/')
    ap.add_argument('--epochs', default=50, type=int, help='number of epochs')
    ap.add_argument('--k_fold', default=5, type=int, help='number of folds')
    ap.add_argument('--val_ratio', default=0.1, type=float, help='validation ratio inside each training fold')
    ap.add_argument('--seed', default=42, type=int, help='random seed')
    ap.add_argument('--eval_interval', default=10, type=int, help='evaluate every N epochs')
    ap.add_argument('--cache_dir', default='cache/subgraph_features', type=str, help='directory used to cache split and bridge-context payloads')
    ap.add_argument('--result_dir', default='', type=str, help='optional directory used to store structured outputs')
    ap.add_argument('--fold_indices', default='', type=str, help='optional 1-based fold indices to run, e.g. "1,2,5"')
    ap.add_argument('--log_tag', default='', type=str, help='optional tag appended to the log filename')
    ap.add_argument('--log_dir', default='logs', type=str, help='directory used to store run logs')
    return ap.parse_args()


def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_fold_indices(raw_value, k_fold):
    text = str(raw_value or '').strip()
    if not text:
        return None
    indices = []
    for chunk in text.split(','):
        token = chunk.strip()
        if not token:
            continue
        fold_id = int(token)
        if fold_id < 1 or fold_id > int(k_fold):
            raise ValueError(f'fold_indices contains out-of-range fold: {fold_id} (valid: 1..{k_fold})')
        indices.append(fold_id)
    if not indices:
        return None
    return sorted(set(indices))


def get_context_feature_dim():
    return 10 + (3 * 3 * 3 + 3) + 4


def initialize_model(data_fold, args):
    in_channels = data_fold.num_node_features

    model = GLCRModel(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        out_channels=args.out_channels,
        dropout=args.dropout1,
    )
    classifier = GLCRClassifier(
        args.out_channels,
        2,
        dropout=args.dropout1,
        subgraph_dim=get_context_feature_dim(),
        latent_mediator_count=getattr(args, 'latent_mediator_count', 16),
        mediator_dropout=getattr(args, 'mediator_dropout', 0.2),
        utility_gate_floor=getattr(args, 'utility_gate_floor', 0.50),
    )
    if hasattr(classifier, 'bridge_prior_scale'):
        classifier.bridge_prior_scale.data.fill_(float(getattr(args, 'bridge_prior_strength', 1.0)))
    return model, classifier


def sample_balanced_links(pos_index, neg_index, seed):
    rng = np.random.default_rng(seed)
    sampled_neg_ids = rng.choice(len(neg_index), size=len(pos_index), replace=False)
    sampled_neg = neg_index[sampled_neg_ids]
    links = np.concatenate([pos_index, sampled_neg], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(pos_index), dtype=np.int64),
            np.zeros(len(sampled_neg), dtype=np.int64),
        ],
        axis=0,
    )
    return links, labels


def get_split_cache_paths(cache_root, dataset_name, seed, k_fold):
    safe_dataset = dataset_name.replace("\\", os.sep).replace("/", os.sep)
    split_root = os.path.join(cache_root, "split_cache", safe_dataset, f"seed_{seed}", f"{k_fold}fold")
    return split_root, os.path.join(split_root, "balanced_links_labels.npz")


def load_or_build_balanced_links(pos_index, neg_index, dataset_name, seed, k_fold, cache_root, logger):
    split_root, balanced_path = get_split_cache_paths(cache_root, dataset_name, seed, k_fold)
    if os.path.exists(balanced_path):
        payload = np.load(balanced_path)
        links = payload["links"].astype(np.int64)
        labels = payload["labels"].astype(np.int64)
        logger.info("Reusing balanced link cache: %s", balanced_path)
        return links, labels, split_root

    os.makedirs(split_root, exist_ok=True)
    logger.info(
        "Building balanced link cache: sampling %d negatives from pool of %d",
        len(pos_index),
        len(neg_index),
    )
    links, labels = sample_balanced_links(pos_index, neg_index, seed)
    np.savez_compressed(balanced_path, links=links, labels=labels)
    logger.info("Saved balanced link cache: %s", balanced_path)
    return links, labels, split_root


def load_or_build_fold_indices(balanced_links, balanced_labels, split_root, seed, k_fold, logger):
    fold_path = os.path.join(split_root, "fold_indices.npz")
    if os.path.exists(fold_path):
        payload = np.load(fold_path)
        folds = []
        for fold_idx in range(k_fold):
            folds.append(
                (
                    payload[f"train_val_idx_{fold_idx}"].astype(np.int64),
                    payload[f"test_idx_{fold_idx}"].astype(np.int64),
                )
            )
        logger.info("Reusing fold split cache: %s", fold_path)
        return folds

    logger.info("Building fold split cache for %d folds", k_fold)
    os.makedirs(split_root, exist_ok=True)
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    folds = []
    save_payload = {}
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(balanced_links, balanced_labels)):
        train_val_idx = train_val_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)
        folds.append((train_val_idx, test_idx))
        save_payload[f"train_val_idx_{fold_idx}"] = train_val_idx
        save_payload[f"test_idx_{fold_idx}"] = test_idx
        logger.info(
            "Fold split cache progress: %d/%d (%.0f%%)",
            fold_idx + 1,
            k_fold,
            ((fold_idx + 1) / k_fold) * 100,
        )
    np.savez_compressed(fold_path, **save_payload)
    logger.info("Saved fold split cache: %s", fold_path)
    return folds


def _load_cached_split_payloads(cache_path):
    payload = np.load(cache_path, allow_pickle=False)
    num_folds = int(payload['num_folds'])
    split_payloads = []
    for fold_idx in range(num_folds):
        fold_payload = {}
        suffix = f'_{fold_idx}'
        for key in payload.files:
            if key == 'num_folds' or not key.endswith(suffix):
                continue
            base_key = key[: -len(suffix)]
            fold_payload[base_key] = payload[key].astype(np.int64)
        split_payloads.append(fold_payload)
    return split_payloads


def _save_split_payloads(cache_path, split_payloads):
    save_payload = {'num_folds': np.array(len(split_payloads), dtype=np.int64)}
    for fold_idx, split_payload in enumerate(split_payloads):
        for key, value in split_payload.items():
            save_payload[f'{key}_{fold_idx}'] = np.asarray(value, dtype=np.int64)
    np.savez_compressed(cache_path, **save_payload)


def load_or_build_fixed_split_payloads(
    balanced_links,
    balanced_labels,
    fold_splits,
    split_root,
    seed,
    val_ratio,
    logger,
):
    cache_name = f'fixed_tvt_val{str(val_ratio).replace(".", "p")}.npz'
    cache_path = os.path.join(split_root, cache_name)
    if os.path.exists(cache_path):
        logger.info('Reusing train/val/test split cache: %s', cache_path)
        return _load_cached_split_payloads(cache_path)

    logger.info('Building fixed train/val/test split cache: %s', cache_path)
    split_payloads = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(fold_splits, start=1):
        train_val_links = balanced_links[train_val_idx]
        train_val_labels = balanced_labels[train_val_idx]
        test_links = balanced_links[test_idx]
        test_labels = balanced_labels[test_idx]
        train_links, train_labels, val_links, val_labels = split_train_val(
            train_val_links,
            train_val_labels,
            val_ratio,
            seed + fold_idx,
        )
        split_payloads.append({
            'train_links': train_links,
            'train_labels': train_labels,
            'val_links': val_links,
            'val_labels': val_labels,
            'test_links': test_links,
            'test_labels': test_labels,
        })
    _save_split_payloads(cache_path, split_payloads)
    logger.info('Saved train/val/test split cache: %s', cache_path)
    return split_payloads


def split_train_val(train_links, train_labels, val_ratio, seed):
    train_links, val_links, train_labels, val_labels = train_test_split(
        train_links,
        train_labels,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed,
    )
    return train_links, train_labels, val_links, val_labels


def to_device_tensors(device, links, labels):
    links_tensor = torch.tensor(links, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return links_tensor, labels_tensor


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return binary_metrics_from_predictions(y_true, y_prob, y_pred)


def binary_metrics_from_predictions(y_true, y_prob, y_pred):
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'aupr': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = float('nan')
    return metrics


def summarize_fold_metrics(fold_metrics):
    summary = {}
    for metric_name in fold_metrics[0].keys():
        values = np.array([metric[metric_name] for metric in fold_metrics], dtype=float)
        if metric_name == 'auc':
            summary[metric_name] = (np.nanmean(values), np.nanstd(values))
        else:
            summary[metric_name] = (np.mean(values), np.std(values))
    return summary


def save_structured_run_outputs(result_dir, fold_metrics, metric_summary, overall_metrics, metadata):
    if not result_dir:
        return
    os.makedirs(result_dir, exist_ok=True)
    fieldnames = ['fold', 'auc', 'aupr', 'f1', 'mcc', 'acc', 'precision', 'recall']
    rows = []
    for fold_idx, metric in enumerate(fold_metrics, start=1):
        rows.append({'fold': fold_idx, **metric})
    rows.append({'fold': 'mean', **{name: metric_summary[name][0] for name in fieldnames[1:]}})
    rows.append({'fold': 'std', **{name: metric_summary[name][1] for name in fieldnames[1:]}})
    with open(os.path.join(result_dir, 'fold_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(result_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'metadata': metadata,
                'fold_mean_std': {
                    metric_name: {'mean': metric_summary[metric_name][0], 'std': metric_summary[metric_name][1]}
                    for metric_name in metric_summary
                },
                'overall_all_test': overall_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def filter_candidate_links(candidate_links, blocked_links):
    blocked = {tuple(edge) for edge in np.asarray(blocked_links, dtype=np.int64).tolist()}
    mask = np.array([tuple(edge) not in blocked for edge in np.asarray(candidate_links, dtype=np.int64).tolist()], dtype=bool)
    return candidate_links[mask]


def sample_negative_links(neg_index, sample_size, seed):
    rng = np.random.default_rng(seed)
    replace = len(neg_index) < sample_size
    selected = rng.choice(len(neg_index), size=sample_size, replace=replace)
    sampled_links = neg_index[selected]
    sampled_labels = np.zeros(len(sampled_links), dtype=np.int64)
    return sampled_links, sampled_labels


def concat_context(context_a, context_b):
    if context_a is None:
        return context_b
    if context_b is None:
        return context_a
    return {key: torch.cat([context_a[key], context_b[key]], dim=0) for key in context_a.keys()}


def index_context(context, indices):
    if context is None:
        return None
    return {key: value[indices] for key, value in context.items()}


def context_to_device(context, device):
    if context is None:
        return None
    return {key: value.to(device) for key, value in context.items()}


def get_monitor_score(metric_dict, metric_name):
    return float(metric_dict[metric_name])


def select_best_threshold(y_true, y_prob, metric_name='mcc'):
    candidates = np.unique(np.concatenate([np.array([0.5], dtype=float), np.asarray(y_prob, dtype=float)]))
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_prob, threshold=best_threshold)
    best_score = get_monitor_score(best_metrics, metric_name)
    for threshold in candidates:
        current_metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        current_score = get_monitor_score(current_metrics, metric_name)
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)
            best_metrics = current_metrics
    return best_threshold, best_metrics


def chunked_logits(
    model,
    classifier,
    data_fold,
    links_tensor,
    context_tensors,
    device,
    args,
    training=False,
    labels_tensor=None,
    return_explain=False,
    return_aux=False,
    perturb_explicit=False,
):
    node_embeddings = model(data_fold)
    padded_embeddings = torch.cat(
        [node_embeddings, node_embeddings.new_zeros((1, node_embeddings.size(1)))],
        dim=0,
    )

    logits_chunks = []
    explain_chunks = []
    aux_chunks = []
    batch_size = max(1, int(getattr(args, 'link_batch_size', 1024)))
    for start in range(0, links_tensor.size(0), batch_size):
        end = min(start + batch_size, links_tensor.size(0))
        link_batch = links_tensor[start:end]
        x_i = node_embeddings[link_batch[:, 0]]
        x_j = node_embeddings[link_batch[:, 1]]
        kwargs = {}
        if context_tensors is not None:
            bridge_ids = context_tensors['bridge_ids'][start:end]
            kwargs['subgraph_emb'] = context_tensors['features'][start:end]
            kwargs['bridge_nodes'] = padded_embeddings[bridge_ids]
            kwargs['bridge_mask'] = context_tensors['bridge_mask'][start:end]
            kwargs['bridge_prior'] = context_tensors['bridge_prior'][start:end]
            kwargs['bridge_stats'] = context_tensors['bridge_stats'][start:end]
            kwargs['bridge_feat'] = context_tensors['bridge_feat'][start:end]
        if return_explain:
            logits, explain = classifier(x_i, x_j, return_explain=True, **kwargs)
            explain_chunks.append(explain)
        elif return_aux:
            logits, aux = classifier(
                x_i,
                x_j,
                return_aux=True,
                perturb_explicit=perturb_explicit,
                **kwargs,
            )
            aux_chunks.append(aux)
        else:
            logits = classifier(x_i, x_j, **kwargs)
        logits_chunks.append(logits)

    logits = torch.cat(logits_chunks, dim=0)
    explain_payload = None
    if return_explain:
        explain_payload = {}
        for key in explain_chunks[0].keys():
            values = [chunk[key] for chunk in explain_chunks if chunk[key] is not None]
            explain_payload[key] = torch.cat(values, dim=0) if values else None
    aux_payload = None
    if return_aux:
        aux_payload = {}
        for key in aux_chunks[0].keys():
            values = [chunk[key] for chunk in aux_chunks if chunk[key] is not None]
            aux_payload[key] = torch.cat(values, dim=0) if values else None
    probs = F.softmax(logits, dim=1)[:, 1]
    return probs, logits, aux_payload, explain_payload


def compute_train_epoch_loss(model, classifier, data_fold, links_tensor, labels_tensor, context_tensors, device, args):
    node_embeddings = model(data_fold)
    padded_embeddings = torch.cat(
        [node_embeddings, node_embeddings.new_zeros((1, node_embeddings.size(1)))],
        dim=0,
    )

    total_count = int(links_tensor.size(0))
    batch_size = max(1, int(getattr(args, 'link_batch_size', 1024)))
    loss_components_total = {
        'ce': 0.0,
        'bridge': 0.0,
        'perturb': 0.0,
    }
    total_loss_value = 0.0

    for start in range(0, total_count, batch_size):
        end = min(start + batch_size, total_count)
        chunk_count = end - start
        link_batch = links_tensor[start:end]
        label_batch = labels_tensor[start:end]
        x_i = node_embeddings[link_batch[:, 0]]
        x_j = node_embeddings[link_batch[:, 1]]

        kwargs = {}
        if context_tensors is not None:
            bridge_ids = context_tensors['bridge_ids'][start:end]
            kwargs['subgraph_emb'] = context_tensors['features'][start:end]
            kwargs['bridge_nodes'] = padded_embeddings[bridge_ids]
            kwargs['bridge_mask'] = context_tensors['bridge_mask'][start:end]
            kwargs['bridge_prior'] = context_tensors['bridge_prior'][start:end]
            kwargs['bridge_stats'] = context_tensors['bridge_stats'][start:end]
            kwargs['bridge_feat'] = context_tensors['bridge_feat'][start:end]

        logits, aux_payload = classifier(
            x_i,
            x_j,
            return_aux=True,
            perturb_explicit=True,
            **kwargs,
        )
        chunk_loss, chunk_components = compute_losses(logits, label_batch, aux_payload, args)
        weight = float(chunk_count) / float(total_count)
        retain_graph = end < total_count
        (chunk_loss * weight).backward(retain_graph=retain_graph)

        total_loss_value += float(chunk_loss.detach().cpu()) * weight
        for key, value in chunk_components.items():
            loss_components_total[key] += float(value) * weight

        del logits, aux_payload, chunk_loss, x_i, x_j, link_batch, label_batch, kwargs

    return total_loss_value, loss_components_total


def evaluate_split(model, classifier, data_fold, links_tensor, labels_tensor, context_tensors, args, threshold=0.5):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        probs, _, _, _ = chunked_logits(
            model,
            classifier,
            data_fold,
            links_tensor,
            context_tensors,
            links_tensor.device,
            args,
            training=False,
        )
        probs = probs.detach().cpu().numpy()
        labels = labels_tensor.detach().cpu().numpy()
    metrics = binary_metrics(labels, probs, threshold=threshold)
    pred_labels = (probs >= threshold).astype(int)
    return probs, labels, pred_labels, metrics


def compute_losses(logits, labels_tensor, aux_payload, args):
    loss = F.cross_entropy(logits, labels_tensor)
    components = {
        'ce': float(loss.detach().cpu()),
        'bridge': 0.0,
        'perturb': 0.0,
    }

    if aux_payload is None:
        raise ValueError('GLCR release training expects auxiliary payloads for the fixed three-term objective.')

    pre_comm_local_seed = aux_payload['pre_comm_local_seed']
    perturbed_logits = aux_payload['perturbed_logits']
    explicit_confidence = aux_payload['explicit_confidence']
    bridge_repr = aux_payload['bridge_repr']
    bridge_confidence = aux_payload['bridge_confidence']

    bridge_target = 1.0 - explicit_confidence.detach()
    confidence_alignment = F.mse_loss(bridge_confidence, bridge_target)
    sufficiency_loss = 0.5 * (
        1.0 - F.cosine_similarity(bridge_repr, pre_comm_local_seed.detach(), dim=1)
    ).pow(2).mean()
    bridge_loss = confidence_alignment + sufficiency_loss
    loss = loss + RELEASE_LAMBDA_BR * bridge_loss
    components['bridge'] = float(bridge_loss.detach().cpu())

    pos_mask = labels_tensor == 1
    if torch.any(pos_mask):
        full_pos = logits[pos_mask, 1]
        perturbed_pos = perturbed_logits[pos_mask, 1]
        perturb_loss = F.relu(perturbed_pos - full_pos).mean()
    else:
        perturb_loss = logits.new_zeros(())
    loss = loss + RELEASE_LAMBDA_PERT * perturb_loss
    components['perturb'] = float(perturb_loss.detach().cpu())

    return loss, components


def clone_state_dict(module):
    return {name: param.detach().cpu().clone() for name, param in module.state_dict().items()}


def main(k_fold, args, logger):
    dataset_folder = 'datasets'
    dataset = Datasets(dataset_folder)
    data, pos_index, neg_index = dataset.process(args.dataset)

    balanced_links, balanced_labels, split_root = load_or_build_balanced_links(
        pos_index,
        neg_index,
        args.dataset,
        args.seed,
        k_fold,
        args.cache_dir,
        logger,
    )
    fold_splits = load_or_build_fold_indices(
        balanced_links,
        balanced_labels,
        split_root,
        args.seed,
        k_fold,
        logger,
    )
    split_payloads = load_or_build_fixed_split_payloads(
        balanced_links,
        balanced_labels,
        fold_splits,
        split_root,
        args.seed,
        args.val_ratio,
        logger,
    )
    balanced_eval_count = len(balanced_links)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)
    logger.info('GLCR mainline context: topk=%s hops=%s', args.bridge_topk, args.subgraph_hops)
    logger.info('Split mode: random')
    logger.info('Negative:positive ratio: 1:1')
    logger.info(
        f'Dataset {args.dataset}: nodes={data.num_nodes}, '
        f'positive_links={len(pos_index)}, negative_links={len(neg_index)}, '
        f'eval_links={balanced_eval_count}'
    )

    fold_metrics = []
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []
    fold_thresholds = []
    selected_folds = parse_fold_indices(getattr(args, 'fold_indices', ''), k_fold)
    if selected_folds is not None:
        logger.info('Running selected folds only: %s', selected_folds)
    executed_fold_ids = []

    for fold, split_payload in enumerate(split_payloads, start=1):
        if selected_folds is not None and fold not in selected_folds:
            continue
        logger.info('')
        logger.info('Fold %d/%d', fold, k_fold)

        train_links = split_payload['train_links']
        train_labels = split_payload['train_labels']
        val_links = split_payload['val_links']
        val_labels = split_payload['val_labels']
        test_links = split_payload['test_links']
        test_labels = split_payload['test_labels']

        blocked_pos = [test_links[test_labels == 1]]
        if np.any(val_labels == 1):
            blocked_pos.append(val_links[val_labels == 1])
        blocked_pos = np.concatenate(blocked_pos, axis=0)

        data_fold, train_links, train_labels = dataset.remove_test_edges(
            data,
            blocked_pos,
            train_links,
            train_labels,
        )

        data_fold_cpu = data_fold.clone()
        data_fold = data_fold.to(device)

        train_links_tensor, train_labels_tensor = to_device_tensors(device, train_links, train_labels)
        val_links_tensor, val_labels_tensor = to_device_tensors(device, val_links, val_labels)
        test_links_tensor, test_labels_tensor = to_device_tensors(device, test_links, test_labels)

        val_ratio_tag = str(args.val_ratio).replace('.', 'p')
        fold_cache_dir = os.path.join(
            args.cache_dir,
            'bridge_context',
            args.dataset,
            'random_balanced_1_1',
            f'seed_{args.seed}',
            f'{k_fold}fold',
            f'val_{val_ratio_tag}',
            f'fold_{fold}',
        )
        train_context = load_or_compute_link_context(
            data_fold_cpu,
            train_links,
            os.path.join(fold_cache_dir, 'train.pt'),
            num_hops=args.subgraph_hops,
            top_k=args.bridge_topk,
        )
        val_context = load_or_compute_link_context(
            data_fold_cpu,
            val_links,
            os.path.join(fold_cache_dir, 'val.pt'),
            num_hops=args.subgraph_hops,
            top_k=args.bridge_topk,
        )
        test_context = load_or_compute_link_context(
            data_fold_cpu,
            test_links,
            os.path.join(fold_cache_dir, 'test.pt'),
            num_hops=args.subgraph_hops,
            top_k=args.bridge_topk,
        )
        train_context_tensors = context_to_device(train_context, device)
        val_context_tensors = context_to_device(val_context, device)
        test_context_tensors = context_to_device(test_context, device)

        model, classifier = initialize_model(data_fold, args)
        model = model.to(device)
        classifier = classifier.to(device)

        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_epoch = 0
        best_val_auc = float('-inf')
        best_model_state = clone_state_dict(model)
        best_classifier_state = clone_state_dict(classifier)

        logger.info(
            f'Train={len(train_links)} Val={len(val_links)} Test={len(test_links)} '
            f'BlockedPositiveEdges={len(blocked_pos)}'
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            classifier.train()
            optimizer.zero_grad(set_to_none=True)

            train_loss, loss_components = compute_train_epoch_loss(
                model,
                classifier,
                data_fold,
                train_links_tensor,
                train_labels_tensor,
                train_context_tensors,
                device,
                args,
            )
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), args.grad_clip)
            optimizer.step()

            should_eval = epoch == 1 or epoch == args.epochs or epoch % args.eval_interval == 0
            if should_eval:
                _, _, _, val_metric = evaluate_split(
                    model,
                    classifier,
                    data_fold,
                    val_links_tensor,
                    val_labels_tensor,
                    val_context_tensors,
                    args,
                )
                logger.info(
                    f'Epoch {epoch:03d} | Loss {train_loss:.4f} '
                    f'(CE {loss_components["ce"]:.4f} Bridge {loss_components["bridge"]:.4f} Per {loss_components["perturb"]:.4f}) | '
                    f'Val AUC {val_metric["auc"]:.4f} | Val ACC {val_metric["acc"]:.4f}'
                )

                if val_metric['auc'] > best_val_auc:
                    best_val_auc = val_metric['auc']
                    best_epoch = epoch
                    best_model_state = clone_state_dict(model)
                    best_classifier_state = clone_state_dict(classifier)

        model.load_state_dict(best_model_state)
        classifier.load_state_dict(best_classifier_state)

        val_prob, val_true, _, _ = evaluate_split(
            model,
            classifier,
            data_fold,
            val_links_tensor,
            val_labels_tensor,
            val_context_tensors,
            args,
        )
        selected_threshold, best_val_metric = select_best_threshold(
            val_true,
            val_prob,
            metric_name=getattr(args, 'threshold_metric', 'mcc'),
        )

        test_prob, test_true, test_pred, test_metric = evaluate_split(
            model,
            classifier,
            data_fold,
            test_links_tensor,
            test_labels_tensor,
            test_context_tensors,
            args,
            threshold=selected_threshold,
        )
        logger.info(
            f'Fold {fold} best_epoch={best_epoch} | '
            f'ValThreshold {selected_threshold:.4f} ({getattr(args, "threshold_metric", "mcc").upper()} {best_val_metric[getattr(args, "threshold_metric", "mcc")]:.4f}) | '
            f'Test AUC {test_metric["auc"]:.4f} | Test AUPR {test_metric["aupr"]:.4f} | '
            f'Test ACC {test_metric["acc"]:.4f}'
        )

        fold_metrics.append(test_metric)
        fold_thresholds.append(float(selected_threshold))
        executed_fold_ids.append(fold)
        all_true_labels.extend(test_true)
        all_pred_probs.extend(test_prob)
        all_pred_labels.extend(test_pred)

    if not fold_metrics:
        raise ValueError('No folds were executed. Please check --fold_indices.')

    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    all_pred_labels = np.array(all_pred_labels)
    overall_metrics = binary_metrics_from_predictions(all_true_labels, all_pred_probs, all_pred_labels)
    metric_summary = summarize_fold_metrics(fold_metrics)

    logger.info('')
    logger.info('%d-Fold Cross Validation Results:', len(executed_fold_ids))
    for fold_id, metric in zip(executed_fold_ids, fold_metrics):
        logger.info(
            f'Fold {fold_id}: '
            f'AUC={metric["auc"]:.4f}, AUPR={metric["aupr"]:.4f}, '
            f'ACC={metric["acc"]:.4f}, F1={metric["f1"]:.4f}'
        )

    logger.info('Average Fold AUC: %.4f', np.nanmean([metric["auc"] for metric in fold_metrics]))
    logger.info('Average Fold AUPR: %.4f', np.mean([metric["aupr"] for metric in fold_metrics]))
    logger.info('Average Fold ACC: %.4f', np.mean([metric["acc"] for metric in fold_metrics]))
    logger.info('%d-Fold Mean +/- Std:', len(executed_fold_ids))
    logger.info('AUC: %.4f +/- %.4f', metric_summary["auc"][0], metric_summary["auc"][1])
    logger.info('AUPR: %.4f +/- %.4f', metric_summary["aupr"][0], metric_summary["aupr"][1])
    logger.info('F1 Score: %.4f +/- %.4f', metric_summary["f1"][0], metric_summary["f1"][1])
    logger.info('MCC: %.4f +/- %.4f', metric_summary["mcc"][0], metric_summary["mcc"][1])
    logger.info('Accuracy: %.4f +/- %.4f', metric_summary["acc"][0], metric_summary["acc"][1])
    logger.info('Precision: %.4f +/- %.4f', metric_summary["precision"][0], metric_summary["precision"][1])
    logger.info('Recall: %.4f +/- %.4f', metric_summary["recall"][0], metric_summary["recall"][1])

    logger.info('')
    logger.info('Performance Metrics on All Test Sets:')
    logger.info('AUC: %.4f', overall_metrics["auc"])
    logger.info('AUPR: %.4f', overall_metrics["aupr"])
    logger.info('F1 Score: %.4f', overall_metrics["f1"])
    logger.info('MCC: %.4f', overall_metrics["mcc"])
    logger.info('Accuracy: %.4f', overall_metrics["acc"])
    logger.info('Precision: %.4f', overall_metrics["precision"])
    logger.info('Recall: %.4f', overall_metrics["recall"])

    save_structured_run_outputs(
        getattr(args, 'result_dir', ''),
        fold_metrics,
        metric_summary,
        overall_metrics,
        {
            'dataset': args.dataset,
            'split_mode': 'random',
            'sampling_strategy': 'balanced_1_to_1',
            'model_release': 'GLCR_release_core2_single_gate',
            'threshold_strategy': getattr(args, 'threshold_strategy', 'validation_auto'),
            'threshold_metric': getattr(args, 'threshold_metric', 'mcc'),
            'fold_thresholds': fold_thresholds,
            'epochs': args.epochs,
            'k_fold': args.k_fold,
            'executed_folds': executed_fold_ids,
        },
    )

    return {
        'fold_metrics': fold_metrics,
        'metric_summary': metric_summary,
        'overall_metrics': overall_metrics,
    }


if __name__ == '__main__':
    args = get_args()
    args = apply_release_defaults(args)
    fix_random_seeds(args.seed)
    logger, log_path = setup_logger(log_dir=args.log_dir, log_tag=args.log_tag)
    try:
        main(args.k_fold, args, logger)
    finally:
        logger.info('Run finished. Log saved to %s', log_path)
