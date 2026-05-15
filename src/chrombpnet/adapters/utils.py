"""Inject, save, and load low-rank adapters on bpnet-lite BPNet modules."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from .layers import LowRankConv1dAdapter, LowRankLinearAdapter


ADAPTER_PRESETS = {
    "profile_only": ["fconv"],
    "count_only": ["linear"],
    "body_only": ["iconv", "rconvs"],
    "heads_only": ["fconv", "linear"],
    "all_conv": ["iconv", "rconvs", "fconv"],
    "all": ["iconv", "rconvs", "fconv", "linear"],
    "rconvs+heads": ["rconvs", "fconv", "linear"],
}

BASE_TARGETS = {"iconv", "rconvs", "fconv", "linear"}


def resolve_adapter_targets(targets):
    """Expand adapter presets and comma/plus-separated target strings."""
    if isinstance(targets, str):
        raw_targets = [targets]
    else:
        raw_targets = list(targets)

    resolved = []
    for target in raw_targets:
        for token in str(target).replace("+", ",").split(","):
            token = token.strip()
            if not token:
                continue
            expanded = ADAPTER_PRESETS.get(token, [token])
            for item in expanded:
                if item not in BASE_TARGETS:
                    valid = sorted(BASE_TARGETS | set(ADAPTER_PRESETS))
                    raise ValueError(f"Unknown adapter target '{item}'. Valid targets: {valid}")
                if item not in resolved:
                    resolved.append(item)
    return resolved


def _freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def _module_device(module):
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _module_specs(accessibility):
    specs = {}
    for name, module in _iter_adapters(accessibility):
        specs[name] = module.manifest()
    return specs


def _iter_adapters(module, prefix=""):
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, (LowRankConv1dAdapter, LowRankLinearAdapter)):
            yield path, child
        yield from _iter_adapters(child, path)


def inject_adapters(
    accessibility,
    targets,
    conv_rank=4,
    linear_rank=4,
    conv_scale=None,
    linear_scale=None,
    dropout=0.0,
):
    """Attach low-rank adapters to selected bpnet-lite BPNet layers."""
    targets = resolve_adapter_targets(targets)
    _freeze_module(accessibility)

    conv_scale = (1.0 / conv_rank) if conv_scale is None else conv_scale
    linear_scale = (1.0 / linear_rank) if linear_scale is None else linear_scale
    injected = []

    if "iconv" in targets:
        accessibility.iconv = LowRankConv1dAdapter(
            accessibility.iconv, rank=conv_rank, scale=conv_scale, dropout=dropout
        )
        injected.append("iconv")

    if "rconvs" in targets:
        for i, conv in enumerate(accessibility.rconvs):
            accessibility.rconvs[i] = LowRankConv1dAdapter(
                conv, rank=conv_rank, scale=conv_scale, dropout=dropout
            )
            injected.append(f"rconvs.{i}")

    if "fconv" in targets:
        accessibility.fconv = LowRankConv1dAdapter(
            accessibility.fconv, rank=conv_rank, scale=conv_scale, dropout=dropout
        )
        injected.append("fconv")

    if "linear" in targets:
        accessibility.linear = LowRankLinearAdapter(
            accessibility.linear, rank=linear_rank, scale=linear_scale, dropout=dropout
        )
        injected.append("linear")

    return {
        "targets": targets,
        "injected": injected,
        "conv_rank": int(conv_rank),
        "linear_rank": int(linear_rank),
        "conv_scale": float(conv_scale),
        "linear_scale": float(linear_scale),
        "dropout": float(dropout),
        "layer_specs": _module_specs(accessibility),
    }


def adapter_parameters(module):
    """Yield trainable adapter parameters."""
    for _name, adapter in _iter_adapters(module):
        yield from adapter.down.parameters()
        yield from adapter.up.parameters()


def count_adapter_parameters(module):
    return sum(parameter.numel() for parameter in adapter_parameters(module))


def assert_deeplift_shap_compatible(module):
    """Validate that injected adapters are affine during DeepLIFT/SHAP.

    Tangermeme's DeepLIFT/SHAP hooks known nonlinear modules and treats all
    other operations as linear. These adapters are compatible when their only
    active operations are Conv1d/Linear layers plus addition/scaling. Dropout is
    allowed in trained adapters, but it must be disabled by eval() before
    attribution.
    """
    for name, adapter in _iter_adapters(module):
        if isinstance(adapter.dropout, torch.nn.Dropout) and adapter.dropout.training:
            raise RuntimeError(
                f"Adapter '{name}' has active dropout. Call model.eval() before "
                "running DeepLIFT/SHAP."
            )
        expected_children = {"base", "down", "up", "dropout"}
        unexpected = set(dict(adapter.named_children())) - expected_children
        if unexpected:
            raise RuntimeError(
                f"Adapter '{name}' has unexpected child modules: {sorted(unexpected)}"
            )
    return True


def adapter_state_dict(module):
    """Return only adapter branch weights, excluding frozen base weights."""
    state = {}
    for name, adapter in _iter_adapters(module):
        for key, value in adapter.state_dict().items():
            if key.startswith("base."):
                continue
            state[f"{name}.{key}"] = value.detach().cpu()
    return state


def _load_adapter_state_dict(module, state_dict):
    missing = []
    unexpected = set(state_dict)
    for name, adapter in _iter_adapters(module):
        prefix = f"{name}."
        local = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        unexpected.difference_update(f"{prefix}{key}" for key in local)
        result = adapter.load_state_dict(local, strict=False)
        missing.extend(f"{prefix}{key}" for key in result.missing_keys if not key.startswith("base."))

    if missing or unexpected:
        raise RuntimeError(
            "Adapter state mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def save_adapter_bundle(path, model, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "manifest": copy.deepcopy(manifest),
        "state_dict": adapter_state_dict(model.accessibility),
    }
    torch.save(bundle, path)


def load_adapter_bundle(model, adapter_path, map_location="cpu"):
    adapter_path = Path(adapter_path)
    bundle = torch.load(adapter_path, weights_only=False, map_location=map_location)
    manifest = bundle["manifest"]
    config = manifest["adapter_config"]
    device = _module_device(model)
    inject_adapters(
        model.accessibility,
        targets=config["targets"],
        conv_rank=config["conv_rank"],
        linear_rank=config["linear_rank"],
        conv_scale=config["conv_scale"],
        linear_scale=config["linear_scale"],
        dropout=config.get("dropout", 0.0),
    )
    model.to(device)
    expected_specs = manifest.get("layer_specs")
    actual_specs = _module_specs(model.accessibility)
    if expected_specs is not None and expected_specs != actual_specs:
        raise RuntimeError(
            f"Adapter manifest does not match injected model layers for {adapter_path}"
        )
    _load_adapter_state_dict(model.accessibility, bundle["state_dict"])
    return model, manifest


def load_model_with_adapter(base_model_path, adapter_path=None, map_location="cpu"):
    model = torch.load(base_model_path, weights_only=False, map_location=map_location)
    if adapter_path is None:
        return model, None
    return load_adapter_bundle(model, adapter_path, map_location=map_location)


class _AccessibilityContainer(torch.nn.Module):
    def __init__(self, accessibility):
        super().__init__()
        self.accessibility = accessibility


def load_accessibility_with_adapter(base_accessibility_path, adapter_path=None, map_location="cpu"):
    accessibility = torch.load(
        base_accessibility_path, weights_only=False, map_location=map_location
    )
    if adapter_path is None:
        return accessibility, None
    container = _AccessibilityContainer(accessibility)
    container, manifest = load_adapter_bundle(
        container, adapter_path, map_location=map_location
    )
    return container.accessibility, manifest
