"""Training loop for AML graph models."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.utils import k_hop_subgraph

import config
from src.gnn_model import AMLDetector

logger = config.setup_logging(__name__)


HYPERPARAMS = {
    "lr": 0.0003,
    "weight_decay": 1e-4,
    "epochs": 400,
    "hidden_dim": 128,
    "heads": 4,
    "dropout": 0.35,
    "gamma_pos": 1.0,
    "gamma_neg": 4.0,
    "clip_asl": 0.05,
    "threshold": 0.35,
    "patience": 50,
    "grad_clip": 1.0,
    "warmup_epochs": 30,
    "batch_size": 1024,
    "num_hops": 1,
}


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss for imbalanced fraud classification."""

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = clip
        self.eps = float(eps)

    def forward(self, logits, targets):
        targets = targets.float()

        probs = torch.sigmoid(logits[:, 1])
        probs_bg = torch.sigmoid(-logits[:, 1])

        if self.clip is not None:
            probs_bg = (probs_bg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs + self.eps) * (1.0 - probs) ** self.gamma_pos
        loss_neg = (1.0 - targets) * torch.log(probs_bg + self.eps) * (1.0 - probs_bg) ** self.gamma_neg

        return (-(loss_pos + loss_neg)).mean()


def _resolve_device(requested_device):
    requested = str(requested_device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested CUDA but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _safe_roc_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float(y_true.mean()) if y_true.size else 0.0
    return float(average_precision_score(y_true, y_prob))


def _binary_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "pr_auc": 0.0,
        }

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
    }


def _threshold_sweep(y_true, y_prob, start=0.1, end=0.7, step=0.01):
    thresholds = np.arange(start, end + 1e-9, step)
    best = {
        "threshold": HYPERPARAMS["threshold"],
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if (f1 > best["f1"]) or (np.isclose(f1, best["f1"]) and precision > best["precision"]):
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def _iter_seed_batches(seed_nodes, batch_size, shuffle):
    if seed_nodes.numel() == 0:
        return

    if shuffle:
        order = torch.randperm(seed_nodes.numel())
    else:
        order = torch.arange(seed_nodes.numel())

    for start in range(0, seed_nodes.numel(), batch_size):
        batch_order = order[start : start + batch_size]
        yield seed_nodes[batch_order]


def _extract_subgraph_batch(data_cpu, seed_nodes, num_hops):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        seed_nodes,
        num_hops=int(num_hops),
        edge_index=data_cpu.edge_index,
        relabel_nodes=True,
        num_nodes=data_cpu.num_nodes,
    )

    x = data_cpu.x[subset]
    edge_attr = getattr(data_cpu, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]

    y_seed = data_cpu.y[seed_nodes]
    return x, edge_index, edge_attr, mapping, y_seed


def _collect_probs(model, data_cpu, eval_nodes, device, threshold, batch_size, num_hops):
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for seed_nodes in _iter_seed_batches(eval_nodes, batch_size=batch_size, shuffle=False):
            x, edge_index, edge_attr, mapping, y_seed = _extract_subgraph_batch(data_cpu, seed_nodes, num_hops)

            x = x.to(device)
            edge_index = edge_index.to(device)
            mapping = mapping.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

            logits = model(x, edge_index, edge_attr=edge_attr)
            probs = torch.softmax(logits[mapping], dim=1)[:, 1]

            y_true = y_seed.detach().cpu().numpy()
            y_prob = probs.detach().cpu().numpy()
            y_true_list.append(y_true)
            y_prob_list.append(y_prob)

    if not y_true_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_prob, y_pred


def train_model(data, device="cpu"):
    """Train AMLDetector with asymmetric focal loss and validation-driven thresholding."""
    config.ensure_dirs()

    resolved_device = _resolve_device(device)

    model = AMLDetector(
        num_features=data.x.shape[1],
        hidden_dim=HYPERPARAMS["hidden_dim"],
        heads=HYPERPARAMS["heads"],
        dropout=HYPERPARAMS["dropout"],
    ).to(resolved_device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 72)
    logger.info("AML training started")
    logger.info("  device: %s", resolved_device)
    logger.info("  cuda_available: %s", torch.cuda.is_available())
    logger.info("  model_parameters: %d", n_params)
    logger.info("  hyperparams: %s", HYPERPARAMS)
    logger.info("=" * 72)

    data_cpu = data.cpu()
    labeled_train = data_cpu.train_mask & (data_cpu.y != -1)
    labeled_val = data_cpu.val_mask & (data_cpu.y != -1)
    labeled_test = data_cpu.test_mask & (data_cpu.y != -1)

    train_nodes = torch.where(labeled_train)[0]
    val_nodes = torch.where(labeled_val)[0]
    test_nodes = torch.where(labeled_test)[0]

    batch_size = int(HYPERPARAMS["batch_size"])
    num_hops = int(HYPERPARAMS["num_hops"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["lr"],
        weight_decay=HYPERPARAMS["weight_decay"],
    )

    total_epochs = int(HYPERPARAMS["epochs"])
    warmup_epochs = int(HYPERPARAMS["warmup_epochs"])

    def lr_lambda(epoch_idx):
        step = epoch_idx + 1
        if step <= warmup_epochs:
            return step / max(float(warmup_epochs), 1.0)
        progress = (step - warmup_epochs) / max(float(total_epochs - warmup_epochs), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = AsymmetricFocalLoss(
        gamma_pos=HYPERPARAMS["gamma_pos"],
        gamma_neg=HYPERPARAMS["gamma_neg"],
        clip=HYPERPARAMS["clip_asl"],
    )

    history = {
        "loss": [],
        "train_f1": [],
        "val_f1": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "lr": [],
    }

    best_state = None
    best_val_f1 = -1.0
    best_val_pr_auc = -1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        batch_losses = []

        for seed_nodes in _iter_seed_batches(train_nodes, batch_size=batch_size, shuffle=True):
            x, edge_index, edge_attr, mapping, y_seed = _extract_subgraph_batch(data_cpu, seed_nodes, num_hops)

            x = x.to(resolved_device)
            edge_index = edge_index.to(resolved_device)
            mapping = mapping.to(resolved_device)
            y_seed = y_seed.to(resolved_device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(resolved_device)

            optimizer.zero_grad()

            logits = model(x, edge_index, edge_attr=edge_attr)
            loss = loss_fn(logits[mapping], y_seed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMS["grad_clip"])
            optimizer.step()

            batch_losses.append(float(loss.item()))

        scheduler.step()

        train_true, train_prob, train_pred = _collect_probs(
            model,
            data_cpu,
            train_nodes,
            resolved_device,
            threshold=HYPERPARAMS["threshold"],
            batch_size=batch_size,
            num_hops=num_hops,
        )
        train_f1 = float(f1_score(train_true, train_pred, zero_division=0)) if train_true.size else 0.0

        val_true, val_prob, val_pred = _collect_probs(
            model,
            data_cpu,
            val_nodes,
            resolved_device,
            threshold=HYPERPARAMS["threshold"],
            batch_size=batch_size,
            num_hops=num_hops,
        )
        val_f1 = float(f1_score(val_true, val_pred, zero_division=0)) if val_true.size else 0.0
        val_pr_auc = _safe_pr_auc(val_true, val_prob)
        val_roc_auc = _safe_roc_auc(val_true, val_prob)

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        history["loss"].append(epoch_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        improved = (val_f1 > best_val_f1) or (np.isclose(val_f1, best_val_f1) and val_pr_auc > best_val_pr_auc)
        if improved:
            best_val_f1 = val_f1
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, config.BEST_MODEL_PATH)
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % config.LOG_INTERVAL == 0:
            logger.info(
                "Epoch %3d/%d | loss=%.4f train_f1=%.4f val_f1=%.4f val_pr_auc=%.4f val_roc_auc=%.4f lr=%.6f",
                epoch,
                total_epochs,
                epoch_loss,
                train_f1,
                val_f1,
                val_pr_auc,
                val_roc_auc,
                float(optimizer.param_groups[0]["lr"]),
            )

        if patience_count >= HYPERPARAMS["patience"]:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    val_true, val_prob, _ = _collect_probs(
        model,
        data_cpu,
        val_nodes,
        resolved_device,
        threshold=HYPERPARAMS["threshold"],
        batch_size=batch_size,
        num_hops=num_hops,
    )
    best_threshold_info = _threshold_sweep(val_true, val_prob, start=0.1, end=0.7, step=0.01)
    best_threshold = float(best_threshold_info["threshold"])

    test_true, test_prob, test_pred = _collect_probs(
        model,
        data_cpu,
        test_nodes,
        resolved_device,
        threshold=best_threshold,
        batch_size=batch_size,
        num_hops=num_hops,
    )
    test_metrics = _binary_metrics(test_true, test_pred, test_prob)

    history["best_threshold"] = best_threshold
    history["best_epoch"] = best_epoch
    history["test_metrics"] = test_metrics

    logger.info("-" * 72)
    logger.info("Best checkpoint epoch: %d", best_epoch)
    logger.info("Best validation F1: %.4f", best_val_f1)
    logger.info("Best validation PR-AUC: %.4f", best_val_pr_auc)
    logger.info("Threshold sweep best threshold: %.2f", best_threshold)
    logger.info(
        "Test metrics @ best threshold | acc=%.4f precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f roc_auc=%.4f",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["pr_auc"],
        test_metrics["roc_auc"],
    )

    return model, history


def get_gnn_predictions(model, data, device="cpu"):
    """Get class predictions and class probabilities for all nodes."""
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    data_cpu = data.cpu()
    all_nodes = torch.arange(data_cpu.num_nodes)
    batch_size = int(HYPERPARAMS["batch_size"])
    num_hops = int(HYPERPARAMS["num_hops"])

    prob_accum = torch.zeros(data_cpu.num_nodes, 2, dtype=torch.float32)

    with torch.no_grad():
        for seed_nodes in _iter_seed_batches(all_nodes, batch_size=batch_size, shuffle=False):
            x, edge_index, edge_attr, mapping, _ = _extract_subgraph_batch(data_cpu, seed_nodes, num_hops)
            x = x.to(resolved_device)
            edge_index = edge_index.to(resolved_device)
            mapping = mapping.to(resolved_device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(resolved_device)

            logits = model(x, edge_index, edge_attr=edge_attr)
            probs = torch.softmax(logits[mapping], dim=1).detach().cpu()
            prob_accum[seed_nodes] = probs

    preds = prob_accum.argmax(dim=1)
    return preds, prob_accum


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from src.data_loader import build_graph, build_pyg_data, load_data, set_seeds
    from src.features import compute_all_features

    set_seeds()
    config.ensure_dirs()

    df, _ = load_data()
    G = build_graph(df)
    features_df = compute_all_features(G, transactions_df=df)
    data, _, _ = build_pyg_data(G, features_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hist = train_model(data, device=device)
    preds, probs = get_gnn_predictions(model, data, device=device)

    print("history_keys", sorted(hist.keys()))
    print("preds", preds.shape, "probs", probs.shape)
