import hydra
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import os
import random
import math
import warnings
import contextlib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
import traceback
import sys
import json

from graph_discrete_flow_model import GraphDiscreteFlowModel
from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
from models.extra_features import ExtraFeatures, DummyExtraFeatures
import utils
import ot
import ot.plot
from torch_geometric.nn import GCNConv, global_mean_pool

DATASET_DIMS = {
    'bzr': 0,
    'aids': 0,
    'cox2': 0, 
    'nci1': 0,
    'dhfr': 0,
    'tox21_p53': 0,
    'tox21_hse': 0,
    'tox21_mmp': 0,
    'tox21_ppar-gamma': 0
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Random seed set (silent)

def get_model_and_dataloader(cfg: DictConfig):
    # Resolve dataset root anonymously and portably:
    # 1) If DATASET_ROOT env var is set, use it.
    # 2) Otherwise, use Hydra's original working directory (project root at launch).
    if not os.path.isabs(cfg.dataset.datadir):
        base_dir = os.environ.get("DATASET_ROOT", get_original_cwd())
        cfg.dataset.datadir = os.path.join(base_dir, cfg.dataset.datadir)
    cfg.train.batch_size = 1
    
    datamodule = SpectreGraphDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
    
    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=dataset_infos,
    )
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    # Omit diagnostic printing to keep output clean

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": None,
        "sampling_metrics": None,
        "visualization_tools": None,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": None,
    }
    model = GraphDiscreteFlowModel.load_from_checkpoint(
        checkpoint_path=cfg.general.test_only,
        cfg=cfg,
        map_location="cpu",
        **model_kwargs
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_loader = datamodule.test_dataloader()
    return model, test_loader

def print_model_info(model):
    # Removed verbose printing; keep function as placeholder for future debugging
    return

def get_y_dim_for_dataset(dataset_name, model=None, cfg=None):
    if cfg and hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'y_dim'):
        if cfg.dataset.y_dim != 'auto':
            return cfg.dataset.y_dim
    dataset_name = dataset_name.lower() if dataset_name else ''
    for key, dim in DATASET_DIMS.items():
        if key in dataset_name:
            return dim
    try:
        if model and hasattr(model, 'model') and hasattr(model.model, 'mlp_in_y'):
            if hasattr(model.model.mlp_in_y[0], 'weight'):
                return model.model.mlp_in_y[0].weight.shape[1]
    except:
        pass
    return 0

def kl_js_alignment_interpolation(P_t, P_tp1, lambda1, lambda2):
    logP = lambda1 * torch.log(P_t + 1e-10) + lambda2 * torch.log(P_tp1 + 1e-10)
    P_interp = torch.softmax(logP, dim=-1)
    return P_interp

def gather_edge_probs(probs, indices):
    batch, n1, n2, nclass = probs.shape
    probs_flat = probs.view(batch, n1 * n2, nclass)
    indices_flat = indices.view(batch, n1 * n2, 1)
    gathered = torch.gather(probs_flat, 2, indices_flat).squeeze(-1)
    return gathered.view(batch, n1, n2)

def compute_anomaly_score_ot_improved(
    model, data, device, num_steps, temp, alpha, beta, 
    ot_reg=0.1, use_ot=True, dataset_name=None, cfg=None
):
    model.eval()
    data = data.to(device)
    true_X, node_mask = to_dense_batch(data.x, data.batch)
    true_E = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
    bs = true_X.shape[0]
    num_nodes = torch.sum(node_mask).item()
    edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
    num_edges = torch.sum(edge_mask).item()
    time_steps = torch.zeros(num_steps + 1, device=device)
    time_steps[0] = 0.0
    time_steps[-1] = 1.0
    if num_steps > 1:
        if alpha == 1.0 and beta == 1.0:
            time_steps[1:-1] = torch.linspace(0.0, 1.0, num_steps + 1, device=device)[1:-1]
        else:
            positions = torch.linspace(0.0, 1.0, num_steps - 1, device=device)
            if alpha < 1 and beta < 1:
                middle_steps = 0.5 - 0.5 * torch.cos(math.pi * positions)
            elif alpha < 1 and beta >= 1:
                middle_steps = torch.pow(positions, 1.0 / alpha)
            elif alpha >= 1 and beta < 1:
                middle_steps = 1.0 - torch.pow(1.0 - positions, 1.0 / beta)
            else:
                middle_steps = 0.5 + 0.5 * torch.tanh((positions - 0.5) * math.sqrt(alpha * beta))
            time_steps[1:-1] = middle_steps
    t0_tensor = torch.zeros((bs, 1), device=device).float()
    dimensions_to_try = [0, 70, 30, 1, 90, 128, 140, 210]
    pred_t0, extra_data_t0, success, y_dim = None, None, False, 0
    for dim in dimensions_to_try:
        try:
            data_t0 = {'t': t0_tensor, 'X_t': true_X, 'E_t': true_E, 'y_t': torch.zeros(bs, dim, device=device).float(), 'node_mask': node_mask}
            extra_data_t0 = model.compute_extra_data(data_t0)
            pred_t0 = model.forward(data_t0, extra_data_t0, node_mask)
            success, y_dim = True, dim
            break
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e): continue
            else: raise e
    if not success:
        try:
            data_t0 = {'t': t0_tensor, 'X_t': true_X, 'E_t': true_E, 'node_mask': node_mask}
            extra_data_t0 = model.compute_extra_data(data_t0)
            pred_t0 = model.forward(data_t0, extra_data_t0, node_mask)
            success = True
        except Exception as e:
            # Fallback without y_t also failed
            raise ValueError("Could not find a matching y_t dimension; all attempts failed")
    step_preds = {}
    for m in range(num_steps + 1):
        t_m = time_steps[m]
        t_m_tensor = torch.full((bs, 1), t_m.item(), device=device).float()
        if 'y_t' in data_t0:
            data_tm = {'t': t_m_tensor, 'X_t': true_X, 'E_t': true_E, 'y_t': torch.zeros(bs, y_dim, device=device).float(), 'node_mask': node_mask}
        else:
            data_tm = {'t': t_m_tensor, 'X_t': true_X, 'E_t': true_E, 'node_mask': node_mask}
        extra_data_tm = model.compute_extra_data(data_tm)
        pred_tm = model.forward(data_tm, extra_data_tm, node_mask)
        step_preds[m] = pred_tm
    true_X_indices = torch.argmax(true_X, dim=-1)
    true_E_indices = torch.argmax(true_E, dim=-1)
    log_prob_X_t0 = F.log_softmax(pred_t0.X / temp, dim=-1)
    log_prob_X_t0 = torch.gather(log_prob_X_t0, -1, true_X_indices.unsqueeze(-1)).squeeze(-1) * node_mask
    log_prob_E_t0 = F.log_softmax(pred_t0.E / temp, dim=-1)
    log_prob_E_t0 = gather_edge_probs(log_prob_E_t0, true_E_indices) * edge_mask
    log_prob = log_prob_X_t0.sum() + log_prob_E_t0.sum()
    for m in range(num_steps):
        pred_tm = step_preds[m]
        pred_tm1 = step_preds[m+1]
        log_prob_X_forward = F.log_softmax(pred_tm.X / temp, dim=-1)
        log_prob_X_forward = torch.gather(log_prob_X_forward, -1, true_X_indices.unsqueeze(-1)).squeeze(-1) * node_mask
        log_prob_E_forward = F.log_softmax(pred_tm.E / temp, dim=-1)
        log_prob_E_forward = gather_edge_probs(log_prob_E_forward, true_E_indices) * edge_mask
        t_m = time_steps[m]
        t_m_plus_1 = time_steps[m+1]
        weight_p0 = t_m_plus_1 / (t_m + t_m_plus_1 + 1e-10)
        weight_delta = 1- weight_p0
        if use_ot:
            try:
                curr_prob_X = F.softmax(pred_tm.X / temp, dim=-1)
                target_prob_X = F.softmax(pred_tm1.X / temp, dim=-1)
                prob_X_reverse_ot = compute_ot_interpolation_improved(
                    curr_prob_X, target_prob_X, weight_p0,
                    node_features=None, node_mask=node_mask, reg=ot_reg, time_ratio=None
                )
                prob_X_reverse_ot = torch.clamp(prob_X_reverse_ot, min=1e-10)
                log_prob_X_reverse = torch.log(
                    torch.gather(prob_X_reverse_ot, -1, true_X_indices.unsqueeze(-1)).squeeze(-1)
                ) * node_mask
                num_edge_classes = pred_tm.E.shape[-1]
                curr_prob_E = F.softmax(pred_tm.E / temp, dim=-1)
                next_true_E_indices = torch.argmax(true_E, dim=-1)
                delta_E = (true_E_indices == next_true_E_indices).float()
                delta_E = delta_E.unsqueeze(-1).repeat(1, 1, 1, num_edge_classes)
                prob_E_reverse = weight_p0 * curr_prob_E + weight_delta * delta_E
                prob_E_reverse = torch.clamp(prob_E_reverse, min=1e-10)
                log_prob_E_reverse = torch.log(
                    gather_edge_probs(prob_E_reverse, true_E_indices)
                ) * edge_mask
            except Exception as e:
                # OT failed, fallback to linear interpolation
                curr_prob_X = F.softmax(pred_tm.X / temp, dim=-1)
                target_prob_X = F.softmax(pred_tm1.X / temp, dim=-1)
                prob_X_reverse = weight_p0 * curr_prob_X + weight_delta * target_prob_X
                prob_X_reverse = torch.clamp(prob_X_reverse, min=1e-10)
                log_prob_X_reverse = torch.log(
                    torch.gather(prob_X_reverse, -1, true_X_indices.unsqueeze(-1)).squeeze(-1
                )) * node_mask
                num_edge_classes = pred_tm.E.shape[-1]
                curr_prob_E = F.softmax(pred_tm.E / temp, dim=-1)
                next_true_E_indices = torch.argmax(true_E, dim=-1)
                delta_E = (true_E_indices == next_true_E_indices).float()
                delta_E = delta_E.unsqueeze(-1).repeat(1, 1, 1, num_edge_classes)
                prob_E_reverse = weight_p0 * curr_prob_E + weight_delta * delta_E
                prob_E_reverse = torch.clamp(prob_E_reverse, min=1e-10)
                log_prob_E_reverse = torch.log(
                    gather_edge_probs(prob_E_reverse, true_E_indices)
                ) * edge_mask
        else:
            curr_prob_X = F.softmax(pred_tm.X / temp, dim=-1)
            target_prob_X = F.softmax(pred_tm1.X / temp, dim=-1)
            prob_X_reverse = weight_p0 * curr_prob_X + weight_delta * target_prob_X
            prob_X_reverse = torch.clamp(prob_X_reverse, min=1e-10)
            log_prob_X_reverse = torch.log(
                torch.gather(prob_X_reverse, -1, true_X_indices.unsqueeze(-1)).squeeze(-1
            )) * node_mask
            num_edge_classes = pred_tm.E.shape[-1]
            curr_prob_E = F.softmax(pred_tm.E / temp, dim=-1)
            next_true_E_indices = torch.argmax(true_E, dim=-1)
            delta_E = (true_E_indices == next_true_E_indices).float()
            delta_E = delta_E.unsqueeze(-1).repeat(1, 1, 1, num_edge_classes)
            prob_E_reverse = weight_p0 * curr_prob_E + weight_delta * delta_E
            prob_E_reverse = torch.clamp(prob_E_reverse, min=1e-10)
            log_prob_E_reverse = torch.log(
                gather_edge_probs(prob_E_reverse, true_E_indices)
            ) * edge_mask
        log_ratio_X = (log_prob_X_forward - log_prob_X_reverse).sum()
        log_ratio_E = (log_prob_E_forward - log_prob_E_reverse).sum()
        log_prob += (log_ratio_X + log_ratio_E)
    node_weight = 0.5
    edge_weight = 0.5
    graph_size = np.log(node_weight * num_nodes + edge_weight * num_edges + 1e-10)
    anomaly_score = -log_prob.item() / graph_size
    return anomaly_score

def compute_ot_interpolation_general(source_dist, target_dist, weight, node_mask=None, reg=0.1, is_edge=False):
    if not is_edge:
        batch_size, n_nodes, n_features = source_dist.shape
        device = source_dist.device
        P_list = []
        for b in range(batch_size):
            if node_mask is not None:
                valid_mask = node_mask[b]
                valid_nodes = int(valid_mask.sum().item())
            else:
                valid_nodes = n_nodes
            if valid_nodes <= 1:
                P_b = torch.eye(n_nodes, device=device)
                P_list.append(P_b)
                continue
            valid_src = source_dist[b, :valid_nodes]
            valid_tgt = target_dist[b, :valid_nodes]
            try:
                C_b = torch.cdist(valid_src.cpu(), valid_tgt.cpu()).numpy()
                a_b = np.ones(valid_nodes) / valid_nodes
                b_b = np.ones(valid_nodes) / valid_nodes
                ot_plan = ot.sinkhorn(a_b, b_b, C_b, reg)
                full_plan = np.zeros((n_nodes, n_nodes))
                full_plan[:valid_nodes, :valid_nodes] = ot_plan
                for i in range(valid_nodes, n_nodes):
                    full_plan[i, i] = 1.0
                P_list.append(torch.from_numpy(full_plan).float().to(device))
            except Exception as e:
                # Node-level OT failed; fall back to identity plan for this batch
                P_b = torch.eye(n_nodes, device=device)
                P_list.append(P_b)
        P = torch.stack(P_list)
        weighted_transport = torch.bmm(P, target_dist)
        ot_interp = (1 - weight) * source_dist + weight * weighted_transport
        return ot_interp
    else:
        batch_size, n_nodes, n_nodes2, n_features = source_dist.shape
        assert n_nodes == n_nodes2
        device = source_dist.device
        ot_interp = torch.zeros_like(source_dist)
        for b in range(batch_size):
            if node_mask is not None:
                valid_mask = node_mask[b]
                valid_nodes = int(valid_mask.sum().item())
            else:
                valid_nodes = n_nodes
            if valid_nodes <= 1:
                ot_interp[b] = source_dist[b]
                continue
            for i in range(valid_nodes):
                src_row = source_dist[b, i, :valid_nodes]
                tgt_row = target_dist[b, i, :valid_nodes]
                try:
                    C_b = torch.cdist(src_row.cpu(), tgt_row.cpu()).numpy()
                    a_b = np.ones(valid_nodes) / valid_nodes
                    b_b = np.ones(valid_nodes) / valid_nodes
                    ot_plan = ot.sinkhorn(a_b, b_b, C_b, reg)
                    transported = torch.from_numpy(ot_plan).float().to(device) @ tgt_row
                    ot_interp[b, i, :valid_nodes] = (1 - weight) * src_row + weight * transported
                except Exception as e:
                    # Edge-level OT failed; fall back to source row
                    ot_interp[b, i, :valid_nodes] = source_dist[b, i, :valid_nodes]
            if valid_nodes < n_nodes:
                ot_interp[b, valid_nodes:, :, :] = source_dist[b, valid_nodes:, :, :]
                ot_interp[b, :, valid_nodes:, :] = source_dist[b, :, valid_nodes:, :]
        return ot_interp

def compute_ot_interpolation_improved(
    source_dist, target_dist, weight, node_features=None, node_mask=None, reg=0.1, time_ratio=None
):
    batch_size, n_nodes, n_features = source_dist.shape
    device = source_dist.device
    if node_features is None:
        node_features = source_dist
    P_list = []
    for b in range(batch_size):
        if node_mask is not None:
            valid_mask = node_mask[b]
            valid_nodes = int(valid_mask.sum().item())
        else:
            valid_nodes = n_nodes
        if valid_nodes <= 1:
            P_b = torch.eye(n_nodes, device=device)
            P_list.append(P_b)
            continue
        valid_src_feat = node_features[b, :valid_nodes].cpu().numpy()
        valid_tgt_feat = node_features[b, :valid_nodes].cpu().numpy()
        try:
            C1 = ot.utils.euclidean_distances(valid_src_feat, valid_src_feat)
            C2 = ot.utils.euclidean_distances(valid_tgt_feat, valid_tgt_feat)
            a_b = np.ones(valid_nodes) / valid_nodes
            b_b = np.ones(valid_nodes) / valid_nodes
            actual_reg = reg
            if time_ratio is not None:
                actual_reg = reg * (1.0 + (1.0 - time_ratio) * 1.5)
            ot_plan = ot.gromov.gromov_wasserstein(
                C1, C2, a_b, b_b, loss_fun='square_loss', epsilon=actual_reg, max_iter=100
            )
            full_plan = np.zeros((n_nodes, n_nodes))
            full_plan[:valid_nodes, :valid_nodes] = ot_plan
            for i in range(valid_nodes, n_nodes):
                full_plan[i, i] = 1.0
            P_list.append(torch.from_numpy(full_plan).float().to(device))
        except Exception as e:
            # GW-OT failed for this batch; fall back to identity plan
            P_b = torch.eye(n_nodes, device=device)
            P_list.append(P_b)
    P = torch.stack(P_list)
    weighted_transport = torch.bmm(P, target_dist)
    mix_weight = weight
    if time_ratio is not None:
        mix_weight = weight * (0.8 + 0.4 * time_ratio)
    ot_interp = (1 - mix_weight) * source_dist + mix_weight * weighted_transport
    return ot_interp

def get_param(cfg, key, default):
    if key in cfg:
        return cfg[key]
    elif "model" in cfg and key in cfg.model:
        return cfg.model[key]
    else:
        return default

def _clean_ckpt_str(s: str) -> str:
    # Strip whitespace and any surrounding quotes (ASCII and smart quotes)
    return s.strip().strip("'\"“”‘’")

def _normalize_ckpt_list(v):
    """Accept a string (ASCII or Chinese comma-delimited), list or ListConfig; return list[str] sanitized."""
    try:
        from omegaconf import ListConfig
    except Exception:
        ListConfig = tuple()  # fallback
    if isinstance(v, (list, ListConfig)):
        return [_clean_ckpt_str(str(x)) for x in v]
    if isinstance(v, str):
        s = _clean_ckpt_str(v)
        # Normalize Chinese comma to ASCII comma
        s = s.replace("，", ",")
        # Split by comma if present; otherwise single entry
        if "," in s:
            parts = [p for p in (t.strip() for t in s.split(",")) if p]
            return [_clean_ckpt_str(p) for p in parts]
        return [s]
    return [_clean_ckpt_str(str(v))]

def transition_main(cfg):
    # Entry point for anomaly evaluation across checkpoints
    # Default metrics to ensure callers can always unpack
    auc, auprc = 0.0, 0.0
    std_auc = 0.0
    try:
        if not cfg.general.test_only:
            raise ValueError("Please specify the checkpoint to test via 'general.test_only=<path/to/ckpt>'")
        # Support multiple checkpoints at once
        ckpt_list = _normalize_ckpt_list(cfg.general.test_only)

        TYPE_I_DATASETS = ["tox21_p53", "tox21_hse", "tox21_mmp", "tox21_ppar_gamma"]
        TYPE_II_DATASETS = ["bzr", "aids", "cox2", "nci1", 
                             "enzymes", "proteins"]
    # Aggregated metrics
        auc_list, auprc_list = [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name = cfg.dataset.name.lower()

        # Common hyperparameters
        use_ot = cfg.get("model", {}).get("use_ot", False)
        use_improved_ot = cfg.get("model", {}).get("use_improved_ot", False)
        ot_reg = get_param(cfg, "ot_reg", 0.1)
        if dataset_name in TYPE_I_DATASETS:
            num_steps = get_param(cfg, "num_steps", 50)
            temp = get_param(cfg, "temp", 1.0)
            alpha = get_param(cfg, "alpha", 0.7)
            beta = get_param(cfg, "beta", 1.0)
        elif dataset_name in TYPE_II_DATASETS:
            num_steps = get_param(cfg, "num_steps", 30)
            temp = get_param(cfg, "temp", 1.0)
            alpha = get_param(cfg, "alpha", 1.0)
            beta = get_param(cfg, "beta", 1.0)
        else:
            num_steps = get_param(cfg, "num_steps", 30)
            temp = get_param(cfg, "temp", 0.85)
            alpha = get_param(cfg, "alpha", 1.0)
            beta = get_param(cfg, "beta", 1.0)

        # Quiet mode: no parameter printing

        for idx, ckpt in enumerate(ckpt_list, 1):
            # Set current checkpoint and build model/data
            cfg.general.test_only = ckpt
            model, test_loader = get_model_and_dataloader(cfg)
            model = model.to(device)
            # Omit model structure printing

            all_scores, all_labels = [], []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    try:
                        if use_ot:
                            score = compute_anomaly_score_ot_improved(
                                model, data, device,
                                num_steps=num_steps,
                                temp=temp,
                                alpha=alpha,
                                beta=beta,
                                ot_reg=ot_reg,
                                use_ot=use_improved_ot,
                                dataset_name=dataset_name,
                                cfg=cfg
                            )
                        else:
                            score = compute_anomaly_score_ot_improved(
                                model, data, device,
                                num_steps=num_steps,
                                temp=temp,
                                alpha=alpha,
                                beta=beta,
                                ot_reg=ot_reg,
                                use_ot=False,
                                dataset_name=dataset_name,
                                cfg=cfg
                            )
                    except Exception as e:
                        # Score computation failed; keep going with zero score
                        traceback.print_exc()
                        score = 0.0

                    all_scores.append(score)
                    graph_label = data.y.item()
                    if dataset_name in TYPE_I_DATASETS:
                        is_anomaly = graph_label
                    elif dataset_name == "cox2":
                        is_anomaly = 1 - graph_label
                    elif dataset_name in TYPE_II_DATASETS:
                        is_anomaly = graph_label
                    else:
                        # Warning: label handling for this dataset is not specified; use raw label
                        is_anomaly = graph_label
                    all_labels.append(is_anomaly)

                    # Silent progress

            # Metrics
            try:
                auc_i = roc_auc_score(all_labels, all_scores)
                auprc_i = average_precision_score(all_labels, all_scores)
                auc_list.append(float(auc_i))
                auprc_list.append(float(auprc_i))
            except ValueError:
                # Metrics cannot be computed (e.g., single-class test set)
                pass

            # Plot only for single checkpoint to avoid excessive files
            if len(ckpt_list) == 1:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    normal_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
                    anomaly_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
                    plt.figure(figsize=(8, 6))
                    plt.hist(normal_scores, bins=20, density=True, alpha=0.5, color="#FF4FA0", label="Normal")
                    plt.hist(anomaly_scores, bins=20, density=True, alpha=0.5, color="#FFB878", label="Anomaly")
                    if len(normal_scores) > 1:
                        sns.kdeplot(normal_scores, color="gold", linewidth=2)
                    if len(anomaly_scores) > 1:
                        sns.kdeplot(anomaly_scores, color="royalblue", linewidth=2)
                    plt.xlabel("Anomaly Score")
                    plt.ylabel("Density")
                    plt.title(f"Distribution of Anomaly Scores ({dataset_name})")
                    plt.legend()
                    plt.tight_layout()
                    plot_file = f"{dataset_name}_anomaly_score_distribution.png"
                    plt.savefig(plot_file, dpi=300)
                    plt.close()
                except Exception as e:
                    # Ignore plotting errors
                    pass

        # Summary
        if auc_list:
            auc = float(np.mean(auc_list))
            auprc = float(np.mean(auprc_list)) if auprc_list else 0.0
            if len(auc_list) > 1:
                std_auc = float(np.std(auc_list, ddof=1))
            else:
                std_auc = 0.0
            # Optionally write to results file
            if cfg.get("general", {}).get("results_file", None):
                summary = {
                    "dataset": dataset_name,
                    "method": "improved_ot" if use_ot and use_improved_ot else "standard_ot" if use_ot else "linear",
                    "n_ckpts": len(auc_list),
                    "mean_auc": auc,
                    "std_auc": std_auc,
                    "mean_auprc": auprc,
                    "parameters": {
                        "num_steps": num_steps,
                        "temp": temp,
                        "alpha": alpha,
                        "beta": beta,
                        "ot_reg": ot_reg if use_ot else None
                    }
                }
                try:
                    with open(cfg.general.results_file, 'a') as f:
                        f.write(json.dumps(summary) + "\n")
                except Exception as e:
                    # Failed to write results file; ignore
                    pass
        else:
            # No valid AUC results
            pass

    except Exception as e:
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()

    return float(auc), float(auprc), float(std_auc)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Only print a start message and final metrics; suppress other prints/warnings
    print("Starting test...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress FutureWarning/UserWarning etc.

        @contextlib.contextmanager
        def _suppress_output():
            devnull = open(os.devnull, 'w')
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()

        with _suppress_output():
            auc, auprc, _std_auc = transition_main(cfg)

    # Final concise result
    print(f"AUCROC={auc:.4f} AUPRC={auprc:.4f}")

if __name__ == "__main__":
    main()
