"""Low-rank adapter utilities for bpnet-lite ChromBPNet models."""

from .layers import LowRankConv1dAdapter, LowRankLinearAdapter
from .utils import (
    ADAPTER_PRESETS,
    adapter_parameters,
    adapter_state_dict,
    count_adapter_parameters,
    inject_adapters,
    load_accessibility_with_adapter,
    load_adapter_bundle,
    load_model_with_adapter,
    resolve_adapter_targets,
    save_adapter_bundle,
)

__all__ = [
    "ADAPTER_PRESETS",
    "LowRankConv1dAdapter",
    "LowRankLinearAdapter",
    "adapter_parameters",
    "adapter_state_dict",
    "count_adapter_parameters",
    "inject_adapters",
    "load_accessibility_with_adapter",
    "load_adapter_bundle",
    "load_model_with_adapter",
    "resolve_adapter_targets",
    "save_adapter_bundle",
]
