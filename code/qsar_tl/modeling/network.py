from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Mapping, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for qsar_tl.modeling.network. "
        "Install the optional ML dependencies, for example: pip install -e .[ml]"
    ) from exc


TOXICITY_BIN_LOGITS_KEY = "__toxicity_bin_logits__"


@dataclass(frozen=True)
class DeepModelConfig:
    numeric_dim: int
    fingerprint_dim: int
    task_heads: tuple[str, ...]
    categorical_cardinalities: Mapping[str, int] = field(default_factory=dict)
    categorical_embedding_dims: Mapping[str, int] = field(default_factory=dict)
    effect_level_numeric_indices: tuple[int, ...] = ()
    descriptor_count: int = 0
    descriptor_encoder_mode: str = "raw"
    descriptor_head_dim: int = 64
    descriptor_group_head_dim: int = 16
    descriptor_group_indices: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    graph_atom_feature_dim: int = 0
    graph_edge_feature_dim: int = 0
    graph_embedding_dim: int = 0
    graph_message_steps: int = 2
    adapter_count: int = 0
    hidden_dims: tuple[int, ...] = (128, 64)
    fusion_mode: str = "concat"
    molecular_hidden_dims: tuple[int, ...] = ()
    context_hidden_dims: tuple[int, ...] = ()
    film_context_hidden_dim: int = 0
    dropout: float = 0.1
    use_molecular_residual: bool = True
    use_adapters: bool = True
    toxicity_bin_count: int = 0
    toxicity_binning_mode: str = "none"
    use_mgkg_residual_adapter: bool = False
    mgkg_residual_adapter_bottleneck: int = 32
    mgkg_residual_adapter_heads: tuple[str, ...] = ()
    mgkg_hierarchical_heads: tuple[str, ...] = ()
    mgkg_hierarchical_head_families: Mapping[str, str] = field(default_factory=dict)
    mgkg_hierarchical_family_scales: Mapping[str, float] = field(default_factory=dict)
    mgkg_hierarchical_task_scales: Mapping[str, float] = field(default_factory=dict)


class EcotoxMultiTaskNetwork(nn.Module):
    """Shared multitask trunk with molecular-signal residual integration."""

    def __init__(self, config: DeepModelConfig) -> None:
        super().__init__()
        if config.numeric_dim < 0 or config.fingerprint_dim < 0:
            raise ValueError("Input dimensions must be non-negative.")
        if not config.task_heads:
            raise ValueError("At least one task head is required.")

        self.config = config
        self.mgkg_residual_adapter_heads = frozenset(config.mgkg_residual_adapter_heads)
        self.effect_level_numeric_indices = tuple(int(index) for index in config.effect_level_numeric_indices)
        invalid_effect_indices = [
            index
            for index in self.effect_level_numeric_indices
            if index < 0 or index >= int(config.numeric_dim)
        ]
        if invalid_effect_indices:
            raise ValueError(
                "effect_level_numeric_indices must point inside the numeric feature matrix: "
                f"{invalid_effect_indices}"
            )
        self.categorical_fields = tuple(sorted(config.categorical_cardinalities))
        self.embeddings = nn.ModuleDict()
        embedding_width = 0
        for field_name in self.categorical_fields:
            cardinality = int(config.categorical_cardinalities[field_name])
            if cardinality <= 0:
                raise ValueError(f"Categorical cardinality for {field_name} must be positive.")
            embedding_dim = int(
                config.categorical_embedding_dims.get(
                    field_name,
                    default_embedding_dim(cardinality),
                )
            )
            self.embeddings[field_name] = nn.Embedding(cardinality, embedding_dim)
            embedding_width += embedding_dim

        self.descriptor_count = min(max(int(config.descriptor_count), 0), int(config.numeric_dim))
        # Historical configurations with descriptor_count=0 treated every numeric
        # feature as molecular input. Preserve that interpretation exactly.
        self.context_numeric_dim = (
            int(config.numeric_dim) - self.descriptor_count
            if self.descriptor_count > 0
            else 0
        )
        self.descriptor_encoder = build_descriptor_encoder(
            mode=config.descriptor_encoder_mode,
            descriptor_count=self.descriptor_count,
            descriptor_head_dim=int(config.descriptor_head_dim),
            descriptor_group_head_dim=int(config.descriptor_group_head_dim),
            descriptor_group_indices=config.descriptor_group_indices,
            dropout=config.dropout,
        )
        self.graph_encoder = (
            SimpleGraphEncoder(
                atom_feature_dim=int(config.graph_atom_feature_dim),
                edge_feature_dim=int(config.graph_edge_feature_dim),
                hidden_dim=int(config.graph_embedding_dim),
                message_steps=int(config.graph_message_steps),
                dropout=float(config.dropout),
            )
            if int(config.graph_atom_feature_dim) > 0 and int(config.graph_embedding_dim) > 0
            else None
        )
        descriptor_dim = self._encoded_descriptor_dim()
        graph_dim = self.graph_encoder.output_dim if self.graph_encoder is not None else 0
        molecular_core_input_dim = descriptor_dim + config.fingerprint_dim + graph_dim
        molecular_input_dim = molecular_core_input_dim + self.context_numeric_dim
        context_input_dim = self.context_numeric_dim + embedding_width
        trunk_input_dim = molecular_input_dim + embedding_width
        if trunk_input_dim <= 0:
            raise ValueError("The model needs at least one input feature.")

        self.fusion_mode = str(config.fusion_mode or "concat").strip().lower()
        if self.fusion_mode == "concat":
            # Keep the historical Sequential name and parameter layout unchanged so
            # legacy checkpoints load without remapping any state-dict keys.
            self.trunk = build_mlp(
                input_dim=trunk_input_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
            head_input_dim = config.hidden_dims[-1] if config.hidden_dims else trunk_input_dim
        elif self.fusion_mode == "additive":
            molecular_dims = tuple(int(value) for value in config.molecular_hidden_dims)
            context_dims = tuple(int(value) for value in config.context_hidden_dims)
            if not molecular_dims or not context_dims:
                raise ValueError("additive fusion requires molecular_hidden_dims and context_hidden_dims.")
            if molecular_dims[-1] != context_dims[-1]:
                raise ValueError("additive molecular and context branches must have the same output width.")
            self.trunk = nn.ModuleDict(
                {
                    "molecular": build_mlp(molecular_core_input_dim, molecular_dims, config.dropout),
                    "context": build_mlp(context_input_dim, context_dims, config.dropout),
                }
            )
            head_input_dim = molecular_dims[-1]
        elif self.fusion_mode == "film":
            molecular_dims = tuple(int(value) for value in config.molecular_hidden_dims)
            film_hidden_dim = int(config.film_context_hidden_dim)
            if not molecular_dims or film_hidden_dim <= 0:
                raise ValueError("film fusion requires molecular_hidden_dims and film_context_hidden_dim.")
            head_input_dim = molecular_dims[-1]
            self.trunk = nn.ModuleDict(
                {
                    "molecular": build_mlp(molecular_core_input_dim, molecular_dims, config.dropout),
                    "context": build_film_context_encoder(
                        context_input_dim,
                        hidden_dim=film_hidden_dim,
                        output_dim=head_input_dim * 2,
                        dropout=config.dropout,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {config.fusion_mode!r}")
        if head_input_dim <= 0:
            raise ValueError("Fusion output dimension must be positive.")
        self.fusion_output_dim = int(head_input_dim)
        self.molecular_core_input_dim = int(molecular_core_input_dim)
        self.context_input_dim = int(context_input_dim)
        self.molecular_residual = (
            nn.Linear(molecular_input_dim, head_input_dim)
            if config.use_molecular_residual and molecular_input_dim > 0
            else None
        )
        self.effect_level_encoder = (
            nn.Sequential(
                nn.Linear(len(self.effect_level_numeric_indices), head_input_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(head_input_dim, head_input_dim),
            )
            if self.effect_level_numeric_indices
            else None
        )
        self.adapters = nn.ModuleList()
        if config.use_adapters and config.adapter_count > 0:
            self.adapters.extend(
                build_adapter(head_input_dim, dropout=config.dropout)
                for _ in range(int(config.adapter_count))
            )
        self.heads = nn.ModuleDict(
            {task_head: nn.Linear(head_input_dim, 1) for task_head in config.task_heads}
        )
        self.mgkg_hierarchical_heads = frozenset(config.mgkg_hierarchical_heads)
        if not self.mgkg_hierarchical_heads.issubset(set(config.task_heads)):
            unknown = sorted(self.mgkg_hierarchical_heads - set(config.task_heads))
            raise ValueError(f"Hierarchical soil heads are not model task heads: {unknown}")
        missing_families = sorted(
            head
            for head in self.mgkg_hierarchical_heads
            if head not in config.mgkg_hierarchical_head_families
        )
        if missing_families:
            raise ValueError(
                "Every hierarchical soil head needs a family mapping: "
                f"{missing_families}"
            )
        self.mgkg_hierarchical_head_families = {
            str(head): str(config.mgkg_hierarchical_head_families[head])
            for head in self.mgkg_hierarchical_heads
        }
        hierarchical_families = tuple(
            sorted(set(self.mgkg_hierarchical_head_families.values()))
        )
        self.mgkg_hierarchical_shared_head = (
            nn.Linear(head_input_dim, 1) if self.mgkg_hierarchical_heads else None
        )
        self.mgkg_hierarchical_family_heads = nn.ModuleDict(
            {family: nn.Linear(head_input_dim, 1) for family in hierarchical_families}
        )
        self.mgkg_hierarchical_family_scales = {
            family: float(config.mgkg_hierarchical_family_scales.get(family, 1.0))
            for family in hierarchical_families
        }
        self.mgkg_hierarchical_task_scales = {
            head: float(config.mgkg_hierarchical_task_scales.get(head, 1.0))
            for head in self.mgkg_hierarchical_heads
        }
        invalid_scales = {
            **{
                f"family:{key}": value
                for key, value in self.mgkg_hierarchical_family_scales.items()
                if not isfinite(value) or not 0.0 <= value <= 1.0
            },
            **{
                f"task:{key}": value
                for key, value in self.mgkg_hierarchical_task_scales.items()
                if not isfinite(value) or not 0.0 <= value <= 1.0
            },
        }
        if invalid_scales:
            raise ValueError(
                "Hierarchical residual contribution scales must be finite values in [0, 1]: "
                f"{invalid_scales}"
            )
        hierarchical_adapter_routes = {
            head in self.mgkg_residual_adapter_heads
            for head in self.mgkg_hierarchical_heads
        }
        if len(hierarchical_adapter_routes) > 1:
            raise ValueError(
                "All hierarchical soil heads must share the same residual-adapter route."
            )
        self.mgkg_hierarchical_uses_residual_adapter = bool(
            hierarchical_adapter_routes == {True}
        )
        if self.mgkg_hierarchical_heads:
            for family_head in self.mgkg_hierarchical_family_heads.values():
                nn.init.zeros_(family_head.weight)
                nn.init.zeros_(family_head.bias)
            for task_head in self.mgkg_hierarchical_heads:
                nn.init.zeros_(self.heads[task_head].weight)
                nn.init.zeros_(self.heads[task_head].bias)
        self.mgkg_residual_adapter = (
            build_zero_initialized_residual_adapter(
                head_input_dim,
                bottleneck_dim=int(config.mgkg_residual_adapter_bottleneck),
                dropout=config.dropout,
            )
            if config.use_mgkg_residual_adapter
            else None
        )
        toxicity_bin_count = max(0, int(config.toxicity_bin_count))
        toxicity_mode = str(config.toxicity_binning_mode or "none").strip().lower()
        self.toxicity_bin_classifier = (
            nn.Linear(head_input_dim, toxicity_bin_count)
            if toxicity_bin_count > 0 and toxicity_mode in {"aux_classification", "ordinal", "soft_expert"}
            else None
        )

    def forward(
        self,
        molecular_numeric: torch.Tensor,
        fingerprint: torch.Tensor,
        categorical_ids: Mapping[str, torch.Tensor] | None = None,
        adapter_ids: torch.Tensor | None = None,
        molecular_graph: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.encode_shared(
            molecular_numeric=molecular_numeric,
            fingerprint=fingerprint,
            categorical_ids=categorical_ids,
            adapter_ids=adapter_ids,
            molecular_graph=molecular_graph,
        )
        mgkg_shared = (
            shared + self.mgkg_residual_adapter(shared)
            if self.mgkg_residual_adapter is not None
            else shared
        )
        outputs: dict[str, torch.Tensor] = {}
        hierarchical_shared = (
            self.mgkg_hierarchical_shared_head(
                mgkg_shared
                if self.mgkg_hierarchical_uses_residual_adapter
                else shared
            ).squeeze(-1)
            if self.mgkg_hierarchical_shared_head is not None
            else None
        )
        for task_head, head in self.heads.items():
            routed_shared = (
                mgkg_shared if task_head in self.mgkg_residual_adapter_heads else shared
            )
            if task_head not in self.mgkg_hierarchical_heads:
                outputs[task_head] = head(routed_shared).squeeze(-1)
                continue
            if hierarchical_shared is None:  # pragma: no cover - guarded at construction
                raise RuntimeError("Hierarchical soil head is missing its shared component.")
            family = self.mgkg_hierarchical_head_families[task_head]
            family_residual = self.mgkg_hierarchical_family_heads[family](
                routed_shared
            ).squeeze(-1)
            task_residual = head(routed_shared).squeeze(-1)
            outputs[task_head] = (
                hierarchical_shared
                + self.mgkg_hierarchical_family_scales[family] * family_residual
                + self.mgkg_hierarchical_task_scales[task_head] * task_residual
            )
        if self.toxicity_bin_classifier is not None:
            outputs[TOXICITY_BIN_LOGITS_KEY] = self.toxicity_bin_classifier(shared)
        return outputs

    def encode_shared(
        self,
        molecular_numeric: torch.Tensor,
        fingerprint: torch.Tensor,
        categorical_ids: Mapping[str, torch.Tensor] | None = None,
        adapter_ids: torch.Tensor | None = None,
        molecular_graph: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        numeric = _ensure_2d_float(molecular_numeric)
        fp = _ensure_2d_float(fingerprint)
        if numeric.shape[0] != fp.shape[0]:
            raise ValueError("molecular_numeric and fingerprint batch sizes differ.")

        descriptor_block, context_numeric = self._encode_numeric_blocks(numeric)
        molecular_core_parts = [descriptor_block, fp]
        legacy_molecular_parts = [descriptor_block]
        if context_numeric.shape[1] > 0:
            legacy_molecular_parts.append(context_numeric)
        legacy_molecular_parts.append(fp)
        graph_embedding = self._encode_graph_for_molecular(
            molecular_graph,
            batch_size=numeric.shape[0],
            device=numeric.device,
            dtype=numeric.dtype,
        )
        if graph_embedding is not None:
            molecular_core_parts.append(graph_embedding)
            legacy_molecular_parts.append(graph_embedding)
        molecular_core = torch.cat(molecular_core_parts, dim=1)
        molecular = torch.cat(legacy_molecular_parts, dim=1)
        context_parts = [context_numeric] if context_numeric.shape[1] > 0 else []
        for field_name in self.categorical_fields:
            ids = _categorical_tensor(
                field_name=field_name,
                batch_size=numeric.shape[0],
                categorical_ids=categorical_ids,
                device=numeric.device,
            )
            context_parts.append(self.embeddings[field_name](ids))

        context = (
            torch.cat(context_parts, dim=1)
            if context_parts
            else numeric.new_zeros((numeric.shape[0], 0))
        )
        if self.fusion_mode == "concat":
            shared = self.trunk(torch.cat([molecular, context[:, self.context_numeric_dim :]], dim=1))
        elif self.fusion_mode == "additive":
            shared = self.trunk["molecular"](molecular_core) + self.trunk["context"](context)
        else:
            molecular_latent = self.trunk["molecular"](molecular_core)
            gamma, beta = self.trunk["context"](context).chunk(2, dim=1)
            shared = molecular_latent * (1.0 + torch.tanh(gamma)) + beta
        if self.molecular_residual is not None:
            shared = shared + self.molecular_residual(molecular)
        if self.effect_level_encoder is not None:
            indices = torch.tensor(self.effect_level_numeric_indices, dtype=torch.long, device=numeric.device)
            shared = shared + self.effect_level_encoder(numeric.index_select(1, indices))
        return self._apply_adapters(shared, adapter_ids)

    def _encode_graph_for_molecular(
        self,
        molecular_graph: Mapping[str, torch.Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.graph_encoder is None:
            return None
        if molecular_graph is None:
            return torch.zeros((batch_size, self.graph_encoder.output_dim), dtype=dtype, device=device)
        graph_payload = {
            key: value.to(device)
            for key, value in molecular_graph.items()
            if torch.is_tensor(value)
        }
        return self.graph_encoder(graph_payload, batch_size=batch_size).to(dtype=dtype)

    def _encoded_descriptor_dim(self) -> int:
        if self.descriptor_count <= 0:
            return int(self.config.numeric_dim)
        return (
            self.descriptor_encoder.output_dim
            if self.descriptor_encoder is not None
            else self.descriptor_count
        )

    def _encoded_numeric_dim(self) -> int:
        return self._encoded_descriptor_dim() + self.context_numeric_dim

    def _encode_numeric_blocks(self, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.descriptor_count <= 0:
            return numeric, numeric[:, :0]
        descriptor_block = numeric[:, : self.descriptor_count]
        context_block = numeric[:, self.descriptor_count :]
        if self.descriptor_encoder is not None:
            descriptor_block = self.descriptor_encoder(descriptor_block)
        return descriptor_block, context_block

    def _encode_numeric_for_molecular(self, numeric: torch.Tensor) -> torch.Tensor:
        descriptor_block, context_block = self._encode_numeric_blocks(numeric)
        if context_block.shape[1] == 0:
            return descriptor_block
        return torch.cat([descriptor_block, context_block], dim=1)

    def _apply_adapters(self, shared: torch.Tensor, adapter_ids: torch.Tensor | None) -> torch.Tensor:
        if not self.adapters:
            return shared
        ids = _adapter_tensor(
            batch_size=shared.shape[0],
            adapter_ids=adapter_ids,
            device=shared.device,
            adapter_count=len(self.adapters),
        )
        adapted = shared.clone()
        for adapter_idx, adapter in enumerate(self.adapters):
            mask = ids == adapter_idx
            if bool(mask.any()):
                adapted[mask] = shared[mask] + adapter(shared[mask])
        return adapted


def default_embedding_dim(cardinality: int) -> int:
    return min(32, max(2, int(sqrt(cardinality)) + 1))


def build_descriptor_encoder(
    *,
    mode: str,
    descriptor_count: int,
    descriptor_head_dim: int,
    descriptor_group_head_dim: int,
    descriptor_group_indices: Mapping[str, tuple[int, ...]],
    dropout: float,
) -> nn.Module | None:
    normalized = str(mode or "raw").strip().lower()
    if descriptor_count <= 0 or normalized in {"", "raw", "none", "off", "identity"}:
        return None
    if normalized in {"dense", "dense_head", "padel_head"}:
        return DenseDescriptorEncoder(
            input_dim=descriptor_count,
            output_dim=max(int(descriptor_head_dim), 1),
            dropout=dropout,
        )
    if normalized in {"prior_clustered", "prior_clustered_heads", "clustered", "clustered_heads"}:
        return PriorClusteredDescriptorEncoder(
            descriptor_count=descriptor_count,
            group_indices=descriptor_group_indices,
            group_output_dim=max(int(descriptor_group_head_dim), 1),
            dropout=dropout,
        )
    allowed = "raw, dense_head, prior_clustered_heads"
    raise ValueError(f"Unsupported descriptor encoder mode '{mode}'. Allowed values: {allowed}")


class DenseDescriptorEncoder(nn.Module):
    """Project a high-dimensional descriptor block before context fusion."""

    def __init__(self, *, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.network(descriptors)


class SimpleGraphEncoder(nn.Module):
    """Small message-passing encoder for cached molecular graphs."""

    def __init__(
        self,
        *,
        atom_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        message_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = max(int(hidden_dim), 1)
        self.message_steps = max(int(message_steps), 1)
        self.atom_proj = nn.Linear(int(atom_feature_dim), self.output_dim)
        self.edge_proj = nn.Linear(max(int(edge_feature_dim), 1), self.output_dim)
        self.update = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, graph: Mapping[str, torch.Tensor], *, batch_size: int) -> torch.Tensor:
        atom_features = graph.get("atom_features")
        if atom_features is None or atom_features.numel() == 0:
            device = next(self.parameters()).device
            return torch.zeros((batch_size, self.output_dim), device=device)
        atom_features = atom_features.float()
        node_state = torch.relu(self.atom_proj(atom_features))
        edge_index = graph.get("edge_index")
        edge_features = graph.get("edge_features")
        if edge_index is not None and edge_features is not None and edge_index.numel() > 0:
            edge_index = edge_index.long()
            edge_features = edge_features.float()
            src = edge_index[0]
            dst = edge_index[1]
            for _ in range(self.message_steps):
                edge_message = node_state.index_select(0, src) + self.edge_proj(edge_features)
                aggregated = torch.zeros_like(node_state)
                aggregated.index_add_(0, dst, edge_message)
                node_state = self.update(torch.cat([node_state, aggregated], dim=1))
        graph_batch = graph.get("graph_batch")
        if graph_batch is None or graph_batch.numel() == 0:
            graph_batch = torch.zeros(node_state.shape[0], dtype=torch.long, device=node_state.device)
        graph_batch = graph_batch.long()
        pooled = torch.zeros((batch_size, self.output_dim), dtype=node_state.dtype, device=node_state.device)
        pooled.index_add_(0, graph_batch.clamp(min=0, max=max(batch_size - 1, 0)), node_state)
        counts = torch.bincount(graph_batch.clamp(min=0, max=max(batch_size - 1, 0)), minlength=batch_size)
        pooled = pooled / counts.to(device=pooled.device, dtype=pooled.dtype).clamp(min=1.0).unsqueeze(1)
        return self.readout(pooled)


class PriorClusteredDescriptorEncoder(nn.Module):
    """Encode descriptor groups with small independent heads.

    The grouping is supplied as descriptor indices from preprocessing config.
    Ungrouped descriptors are kept in a final auxiliary head so the clustered
    branch can be audited without silently discarding input dimensions.
    """

    def __init__(
        self,
        *,
        descriptor_count: int,
        group_indices: Mapping[str, tuple[int, ...]],
        group_output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.group_names: list[str] = []
        self.group_indices: list[torch.Tensor] = []
        self.heads = nn.ModuleList()
        used: set[int] = set()
        for group_name in sorted(group_indices):
            indices = tuple(
                sorted(
                    {
                        int(index)
                        for index in group_indices[group_name]
                        if 0 <= int(index) < int(descriptor_count)
                    }
                )
            )
            if not indices:
                continue
            self.group_names.append(str(group_name))
            self.group_indices.append(torch.tensor(indices, dtype=torch.long))
            self.heads.append(_descriptor_group_head(len(indices), int(group_output_dim), dropout))
            used.update(indices)
        ungrouped = tuple(index for index in range(int(descriptor_count)) if index not in used)
        if ungrouped:
            self.group_names.append("__ungrouped__")
            self.group_indices.append(torch.tensor(ungrouped, dtype=torch.long))
            self.heads.append(_descriptor_group_head(len(ungrouped), int(group_output_dim), dropout))
        if not self.heads:
            raise ValueError("prior_clustered descriptor encoder needs at least one non-empty descriptor group.")
        self.output_dim = len(self.heads) * int(group_output_dim)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        encoded = []
        for indices, head in zip(self.group_indices, self.heads):
            group_indices = indices.to(device=descriptors.device)
            encoded.append(head(descriptors.index_select(1, group_indices)))
        return torch.cat(encoded, dim=1)


def _descriptor_group_head(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = max(output_dim, min(max(input_dim, 4), 64))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("Hidden dimensions must be positive.")
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


def build_film_context_encoder(
    input_dim: int,
    *,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a FiLM generator whose modulation starts at the molecular-only function."""

    if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("FiLM context encoder dimensions must be positive.")
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    output_layer = nn.Linear(hidden_dim, output_dim)
    nn.init.zeros_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return nn.Sequential(*layers)


def build_adapter(hidden_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)


def build_zero_initialized_residual_adapter(
    hidden_dim: int,
    *,
    bottleneck_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a conservative target-domain adapter that starts as an identity residual."""

    bottleneck = max(1, min(int(bottleneck_dim), int(hidden_dim)))
    layers: list[nn.Module] = [
        nn.Linear(hidden_dim, bottleneck),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    output_layer = nn.Linear(bottleneck, hidden_dim)
    nn.init.zeros_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)
    layers.append(output_layer)
    return nn.Sequential(*layers)


def _ensure_2d_float(values: torch.Tensor) -> torch.Tensor:
    tensor = values.float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _categorical_tensor(
    *,
    field_name: str,
    batch_size: int,
    categorical_ids: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    if categorical_ids is None or field_name not in categorical_ids:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    ids = categorical_ids[field_name].to(device=device, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.repeat(batch_size)
    if ids.ndim != 1:
        raise ValueError(f"Categorical ids for {field_name} must be a 1D tensor.")
    if ids.shape[0] != batch_size:
        raise ValueError(
            f"Categorical ids for {field_name} have batch size {ids.shape[0]}, "
            f"expected {batch_size}."
        )
    return ids


def _adapter_tensor(
    *,
    batch_size: int,
    adapter_ids: torch.Tensor | None,
    device: torch.device,
    adapter_count: int,
) -> torch.Tensor:
    if adapter_ids is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    ids = adapter_ids.to(device=device, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.repeat(batch_size)
    if ids.ndim != 1:
        raise ValueError("adapter_ids must be a 1D tensor.")
    if ids.shape[0] != batch_size:
        raise ValueError(f"adapter_ids have batch size {ids.shape[0]}, expected {batch_size}.")
    return ids.clamp(min=0, max=max(adapter_count - 1, 0))
