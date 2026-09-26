from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskHead:
    name: str
    group: str
    target: str
    weight: float
    loss_mask: str = "always"


DEFAULT_TASK_HEADS = [
    TaskHead("ECx_Mortality", "main", "target", 1.0),
    TaskHead("ECx_Growth", "main", "target", 1.0),
    TaskHead("ECx_Reproduction", "main", "target", 1.0),
    TaskHead("NOEC_Growth", "main", "target", 1.0),
    TaskHead("LOEC_Reproduction", "main", "target", 1.0),
    TaskHead("logBCF", "bioaccumulation_aux", "logBCF", 0.3, "observed_only"),
    TaskHead("logBAF", "bioaccumulation_aux", "logBAF", 0.3, "observed_only"),
    TaskHead("oral", "oral_aux", "neg_log10_mg_kg_bw", 0.3),
]

