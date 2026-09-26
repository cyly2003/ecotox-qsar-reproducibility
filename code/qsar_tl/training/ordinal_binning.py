from __future__ import annotations

from typing import Any


def ordinal_softmax_loss(logits: Any, targets: Any) -> Any:
    """Distance-aware ordinal loss for toxicity-bin logits.

    The existing toxicity-bin head emits one logit per ordered class. This loss
    keeps that head shape and compares predicted and target cumulative
    distributions, so errors that jump across several bins are penalized more
    than adjacent-bin errors.
    """

    import torch

    labels = targets.to(device=logits.device, dtype=torch.long)
    mask = labels >= 0
    if not bool(mask.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    active_logits = logits[mask]
    active_labels = labels[mask]
    class_count = int(active_logits.shape[1])
    if class_count <= 1:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    probabilities = torch.nn.functional.softmax(active_logits, dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)
    target_one_hot = torch.nn.functional.one_hot(active_labels, num_classes=class_count).to(
        dtype=logits.dtype,
        device=logits.device,
    )
    target_cdf = target_one_hot.cumsum(dim=1)
    return torch.mean(torch.sum((predicted_cdf - target_cdf).pow(2), dim=1) / class_count)
