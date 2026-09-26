from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from qsar_tl.modeling.network import TOXICITY_BIN_LOGITS_KEY
from qsar_tl.training.censored_loss import censored_hinge_loss
from qsar_tl.training.ordinal_binning import ordinal_softmax_loss

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for qsar_tl.training.deep_train. "
        "Install the optional ML dependencies, for example: pip install -e .[ml]"
    ) from exc


@dataclass(frozen=True)
class DeepTrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    huber_delta: float = 1.0
    mse_loss_weight: float = 0.0
    task_weights: Mapping[str, float] = field(default_factory=dict)
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = None
    scheduler: str = "none"
    device: str = "cpu"
    seed: int = 42
    num_workers: int = 0
    toxicity_bin_loss_weight: float = 0.0
    toxicity_binning_mode: str = "aux_classification"
    censored_loss_weight: float = 0.0
    censored_loss_margin: float = 0.0


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    mean_loss: float
    samples: int
    task_loss: dict[str, float]


@dataclass(frozen=True)
class TrainingHistory:
    epochs: tuple[EpochMetrics, ...]


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_aggregated_task_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    molecular_numeric = _stack_float_matrix(batch, "molecular_numeric")
    fingerprint = _stack_float_matrix(batch, "fingerprint")
    target_value = torch.tensor(
        [float(sample["target_value"]) for sample in batch],
        dtype=torch.float32,
    )
    sample_weight = torch.tensor(
        [float(sample.get("sample_weight", 1.0) or 1.0) for sample in batch],
        dtype=torch.float32,
    )
    toxicity_bin_index = torch.tensor(
        [int(sample.get("toxicity_bin_index", -1) if sample.get("toxicity_bin_index", -1) is not None else -1) for sample in batch],
        dtype=torch.long,
    )
    censored_direction_id = torch.tensor(
        [int(sample.get("censored_direction_id", 0) or 0) for sample in batch],
        dtype=torch.long,
    )
    adapter_id = torch.tensor(
        [int(sample.get("adapter_id", 0) or 0) for sample in batch],
        dtype=torch.long,
    )
    task_head = [str(sample["task_head"]) for sample in batch]

    categorical_fields = sorted(
        {
            field_name
            for sample in batch
            for field_name in dict(sample.get("categorical_ids", {}))
        }
    )
    categorical_ids = {
        field_name: torch.tensor(
            [
                int(dict(sample.get("categorical_ids", {})).get(field_name, 0))
                for sample in batch
            ],
            dtype=torch.long,
        )
        for field_name in categorical_fields
    }

    return {
        "molecular_numeric": molecular_numeric,
        "fingerprint": fingerprint,
        "molecular_graph": _collate_molecular_graphs(batch),
        "categorical_ids": categorical_ids,
        "adapter_id": adapter_id,
        "task_head": task_head,
        "target_value": target_value,
        "sample_weight": sample_weight,
        "toxicity_bin_index": toxicity_bin_index,
        "censored_direction_id": censored_direction_id,
        "split_part": [str(sample.get("split_part", "")) for sample in batch],
        "medium_domain": [str(sample.get("medium_domain", "")) for sample in batch],
    }


def _collate_molecular_graphs(batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor] | None:
    graphs = [sample.get("molecular_graph") for sample in batch]
    if not any(graphs):
        return None

    atom_dim, edge_dim = _graph_feature_dims(graphs)
    atom_rows: list[list[float]] = []
    edge_rows: list[list[float]] = []
    edge_pairs: list[tuple[int, int]] = []
    graph_batch: list[int] = []
    atom_offset = 0
    for graph_idx, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            atom_features = [[0.0] * atom_dim]
            edge_index = []
            edge_features = []
        else:
            atom_features = _coerce_graph_matrix(graph.get("atom_features"), width=atom_dim)
            edge_index = graph.get("edge_index") or []
            edge_features = _coerce_graph_matrix(graph.get("edge_features"), width=edge_dim)
            if not atom_features:
                atom_features = [[0.0] * atom_dim]
                edge_index = []
                edge_features = []
        for atom in atom_features:
            atom_rows.append(atom)
            graph_batch.append(graph_idx)
        for pair, features in zip(edge_index, edge_features):
            if not isinstance(pair, Sequence) or len(pair) < 2:
                continue
            edge_pairs.append((int(pair[0]) + atom_offset, int(pair[1]) + atom_offset))
            edge_rows.append(features)
        atom_offset += len(atom_features)

    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_feature_tensor = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_feature_tensor = torch.empty((0, edge_dim), dtype=torch.float32)
    return {
        "atom_features": torch.tensor(atom_rows, dtype=torch.float32),
        "edge_index": edge_index_tensor,
        "edge_features": edge_feature_tensor,
        "graph_batch": torch.tensor(graph_batch, dtype=torch.long),
        "batch_size": torch.tensor(len(batch), dtype=torch.long),
    }


def _graph_feature_dims(graphs: Sequence[Any]) -> tuple[int, int]:
    atom_dim = 5
    edge_dim = 6
    for graph in graphs:
        if not isinstance(graph, Mapping):
            continue
        atom_features = graph.get("atom_features") or []
        if atom_features:
            atom_dim = len(atom_features[0])
        edge_features = graph.get("edge_features") or []
        if edge_features:
            edge_dim = len(edge_features[0])
        if atom_features and edge_features:
            break
    return int(atom_dim), int(edge_dim)


def _coerce_graph_matrix(value: Any, *, width: int) -> list[list[float]]:
    if not value:
        return []
    rows: list[list[float]] = []
    for row in value:
        values = [float(item) for item in list(row)[:width]]
        if len(values) < width:
            values.extend([0.0] * (width - len(values)))
        rows.append(values)
    return rows


def train_model(
    model: nn.Module,
    dataset: Any,
    config: DeepTrainingConfig | None = None,
) -> TrainingHistory:
    train_config = config or DeepTrainingConfig()
    if train_config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if train_config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_torch_seed(train_config.seed)
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = build_optimizer(model.parameters(), train_config)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_aggregated_task_batch,
        **dataloader_runtime_options(train_config.device, train_config.num_workers),
    )

    metrics: list[EpochMetrics] = []
    for epoch in range(1, train_config.epochs + 1):
        metrics.append(
            train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                config=train_config,
                epoch=epoch,
                device=device,
            )
        )
    return TrainingHistory(epochs=tuple(metrics))


def dataloader_runtime_options(
    device: torch.device | str,
    num_workers: int,
    *,
    persistent_workers: bool = False,
) -> dict[str, Any]:
    """Return protocol-invariant loader options for faster host/device transfer."""
    worker_count = max(0, int(num_workers))
    options: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": torch.device(device).type == "cuda",
    }
    if worker_count > 0:
        options["prefetch_factor"] = 2
        if persistent_workers:
            options["persistent_workers"] = True
    return options


def graph_to_device(graph: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if graph is None:
        return None
    return {
        key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
        for key, value in graph.items()
    }


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    config: DeepTrainingConfig,
    epoch: int = 1,
    device: torch.device | None = None,
) -> EpochMetrics:
    model.train()
    target_device = device or torch.device(config.device)
    loss_fn = nn.HuberLoss(delta=config.huber_delta, reduction="mean")

    total_weighted_loss = 0.0
    total_samples = 0
    task_loss_sum: dict[str, float] = {}
    task_counts: dict[str, int] = {}

    for batch in dataloader:
        molecular_numeric = batch["molecular_numeric"].to(target_device, non_blocking=True)
        fingerprint = batch["fingerprint"].to(target_device, non_blocking=True)
        molecular_graph = graph_to_device(batch.get("molecular_graph"), target_device, non_blocking=True)
        categorical_ids = {
            field_name: ids.to(target_device, non_blocking=True)
            for field_name, ids in batch["categorical_ids"].items()
        }
        targets = batch["target_value"].to(target_device, non_blocking=True)
        adapter_ids = batch.get("adapter_id")
        adapter_ids = adapter_ids.to(target_device, non_blocking=True) if adapter_ids is not None else None
        task_heads = list(batch["task_head"])
        toxicity_bin_index = batch.get("toxicity_bin_index")
        toxicity_bin_index = toxicity_bin_index.to(target_device, non_blocking=True) if toxicity_bin_index is not None else None
        censored_direction_id = batch.get("censored_direction_id")
        censored_direction_id = censored_direction_id.to(target_device, non_blocking=True) if censored_direction_id is not None else None

        optimizer.zero_grad(set_to_none=True)
        model_kwargs = {"adapter_ids": adapter_ids}
        if molecular_graph is not None:
            model_kwargs["molecular_graph"] = molecular_graph
        outputs = model(
            molecular_numeric=molecular_numeric,
            fingerprint=fingerprint,
            categorical_ids=categorical_ids,
            **model_kwargs,
        )
        loss = masked_multitask_huber_loss(
            outputs=outputs,
            targets=targets,
            task_heads=task_heads,
            task_weights=config.task_weights,
            loss_fn=loss_fn,
            toxicity_bin_index=toxicity_bin_index,
            toxicity_bin_loss_weight=config.toxicity_bin_loss_weight,
            toxicity_binning_mode=config.toxicity_binning_mode,
            censored_direction_id=censored_direction_id,
            censored_loss_weight=config.censored_loss_weight,
            censored_loss_margin=config.censored_loss_margin,
        )
        loss.backward()
        if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
        optimizer.step()

        batch_size = len(task_heads)
        total_weighted_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        _accumulate_task_losses(
            outputs=outputs,
            targets=targets,
            task_heads=task_heads,
            task_loss_sum=task_loss_sum,
            task_counts=task_counts,
            loss_fn=loss_fn,
        )

    mean_loss = total_weighted_loss / max(total_samples, 1)
    task_loss = {
        task_head: task_loss_sum[task_head] / task_counts[task_head]
        for task_head in sorted(task_loss_sum)
    }
    return EpochMetrics(
        epoch=epoch,
        mean_loss=mean_loss,
        samples=total_samples,
        task_loss=task_loss,
    )


def masked_multitask_huber_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    task_heads: Sequence[str],
    task_weights: Mapping[str, float],
    loss_fn: nn.Module,
    toxicity_bin_index: torch.Tensor | None = None,
    toxicity_bin_loss_weight: float = 0.0,
    toxicity_binning_mode: str = "aux_classification",
    censored_direction_id: torch.Tensor | None = None,
    censored_loss_weight: float = 0.0,
    censored_loss_margin: float = 0.0,
) -> torch.Tensor:
    device = targets.device
    losses: list[torch.Tensor] = []
    censored_mask = (
        torch.zeros(targets.shape[0], dtype=torch.bool, device=device)
        if censored_direction_id is None
        else censored_direction_id.to(device=device, dtype=torch.long) != 0
    )
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        task_mask = torch.tensor(
            [head == task_head for head in task_heads],
            dtype=torch.bool,
            device=device,
        )
        mask = task_mask & ~censored_mask
        if not bool(mask.any()):
            continue
        weight = float(task_weights.get(task_head, 1.0))
        losses.append(weight * loss_fn(outputs[task_head][mask], targets[mask]))
    if not losses:
        regression_loss = torch.zeros((), dtype=targets.dtype, device=device)
        weight_sum = 1.0
    else:
        weight_sum = sum(
            float(task_weights.get(task_head, 1.0))
            for task_head in sorted(set(head for head, is_censored in zip(task_heads, censored_mask.tolist()) if not is_censored))
        )
        regression_loss = torch.stack(losses).sum() / max(weight_sum, 1e-12)
    censored_loss = multitask_censored_hinge_loss(
        outputs,
        targets=targets,
        task_heads=task_heads,
        censored_direction_id=censored_direction_id,
        margin=censored_loss_margin,
    )
    aux_loss = toxicity_bin_classification_loss(
        outputs,
        toxicity_bin_index=toxicity_bin_index,
        mode=toxicity_binning_mode,
    )
    total_loss = regression_loss
    if aux_loss is not None and float(toxicity_bin_loss_weight) > 0:
        total_loss = total_loss + float(toxicity_bin_loss_weight) * aux_loss
    if censored_loss is not None and float(censored_loss_weight) > 0:
        total_loss = total_loss + float(censored_loss_weight) * censored_loss
    return total_loss


def multitask_censored_hinge_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    targets: torch.Tensor,
    task_heads: Sequence[str],
    censored_direction_id: torch.Tensor | None,
    margin: float = 0.0,
) -> torch.Tensor | None:
    if censored_direction_id is None:
        return None
    device = targets.device
    direction_ids = censored_direction_id.to(device=device, dtype=torch.long)
    losses: list[torch.Tensor] = []
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        mask = torch.tensor([head == task_head for head in task_heads], dtype=torch.bool, device=device)
        mask = mask & (direction_ids != 0)
        if bool(mask.any()):
            losses.append(censored_hinge_loss(outputs[task_head][mask], targets[mask], direction_ids[mask], margin=margin))
    if not losses:
        return None
    return torch.stack(losses).mean()


def toxicity_bin_classification_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    toxicity_bin_index: torch.Tensor | None,
    mode: str = "aux_classification",
) -> torch.Tensor | None:
    logits = outputs.get(TOXICITY_BIN_LOGITS_KEY)
    if logits is None or toxicity_bin_index is None:
        return None
    targets = toxicity_bin_index.to(device=logits.device, dtype=torch.long)
    mask = targets >= 0
    if not bool(mask.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    if str(mode or "").strip().lower() == "ordinal":
        return ordinal_softmax_loss(logits, targets)
    return torch.nn.functional.cross_entropy(logits[mask], targets[mask])


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: DeepTrainingConfig,
) -> torch.optim.Optimizer:
    optimizer_name = (config.optimizer or "adamw").strip().lower()
    params = list(parameters)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=float(config.learning_rate))
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'. Use 'adamw' or 'adam'.")


def _accumulate_task_losses(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    task_heads: Sequence[str],
    task_loss_sum: dict[str, float],
    task_counts: dict[str, int],
    loss_fn: nn.Module,
) -> None:
    for task_head in sorted(set(task_heads)):
        mask = torch.tensor(
            [head == task_head for head in task_heads],
            dtype=torch.bool,
            device=targets.device,
        )
        count = int(mask.sum().detach().cpu())
        if count == 0:
            continue
        value = float(loss_fn(outputs[task_head][mask].detach(), targets[mask]).cpu())
        task_loss_sum[task_head] = task_loss_sum.get(task_head, 0.0) + value * count
        task_counts[task_head] = task_counts.get(task_head, 0) + count


def _stack_float_matrix(batch: Sequence[Mapping[str, Any]], field_name: str) -> torch.Tensor:
    rows = [list(sample.get(field_name, [])) for sample in batch]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"All '{field_name}' rows in a batch must have the same width.")
    if width == 0:
        return torch.empty((len(rows), 0), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)
