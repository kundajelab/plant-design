import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chrombpnet.adapters import (  # noqa: E402
    LowRankConv1dAdapter,
    LowRankLinearAdapter,
    adapter_parameters,
    assert_deeplift_shap_compatible,
    count_adapter_parameters,
    inject_adapters,
    load_adapter_bundle,
    save_adapter_bundle,
)


class TinyAccessibility(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.iconv = torch.nn.Conv1d(4, 3, kernel_size=3, padding=1)
        self.rconvs = torch.nn.ModuleList(
            [torch.nn.Conv1d(3, 3, kernel_size=3, padding=1)]
        )
        self.fconv = torch.nn.Conv1d(3, 1, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, X):
        X = torch.relu(self.iconv(X))
        X = X + torch.relu(self.rconvs[0](X))
        profile = self.fconv(X)
        counts = self.linear(X.mean(dim=-1))
        return profile, counts


class TinyChromBPNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.accessibility = TinyAccessibility()


def test_conv_adapter_is_identity_at_initialization():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(4, 5, kernel_size=3, padding=1)
    adapter = LowRankConv1dAdapter(conv, rank=2, scale=0.5)
    X = torch.randn(2, 4, 11)

    torch.testing.assert_close(adapter(X), conv(X))


def test_linear_adapter_is_identity_at_initialization():
    torch.manual_seed(0)
    linear = torch.nn.Linear(5, 2)
    adapter = LowRankLinearAdapter(linear, rank=2, scale=0.5)
    X = torch.randn(7, 5)

    torch.testing.assert_close(adapter(X), linear(X))


def test_adapters_are_affine_in_eval_mode():
    torch.manual_seed(0)
    adapter = LowRankConv1dAdapter(
        torch.nn.Conv1d(4, 5, kernel_size=3, padding=1),
        rank=2,
        scale=0.5,
        dropout=0.25,
    )
    for parameter in adapter.parameters():
        if parameter.requires_grad:
            torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    adapter.eval()

    X = torch.randn(2, 4, 11)
    R = torch.randn(2, 4, 11)
    mixed = 0.3 * X + 0.7 * R

    torch.testing.assert_close(adapter(mixed), 0.3 * adapter(X) + 0.7 * adapter(R))


def test_deeplift_compatibility_requires_dropout_eval_mode():
    model = TinyChromBPNet()
    inject_adapters(model.accessibility, targets=["heads_only"], dropout=0.25)

    with pytest.raises(RuntimeError, match="dropout"):
        assert_deeplift_shap_compatible(model.accessibility)

    model.eval()
    assert assert_deeplift_shap_compatible(model.accessibility)


def test_load_adapter_bundle_preserves_base_model_device(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyChromBPNet()
    config = inject_adapters(model.accessibility, targets=["heads_only"])
    manifest = {
        "format_version": 1,
        "adapter_config": {
            "targets": config["targets"],
            "conv_rank": config["conv_rank"],
            "linear_rank": config["linear_rank"],
            "conv_scale": config["conv_scale"],
            "linear_scale": config["linear_scale"],
            "dropout": config["dropout"],
        },
        "layer_specs": config["layer_specs"],
    }
    path = tmp_path / "adapter.pt"
    save_adapter_bundle(path, model, manifest)

    base = TinyChromBPNet().to(device)
    loaded, _ = load_adapter_bundle(base, path, map_location=device)

    devices = {parameter.device for parameter in loaded.parameters()}
    assert devices == {device}


def test_only_adapter_parameters_require_grad():
    model = TinyChromBPNet()
    inject_adapters(
        model.accessibility,
        targets=["all"],
        conv_rank=2,
        linear_rank=1,
        conv_scale=0.5,
        linear_scale=1.0,
    )

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(".down." in name or ".up." in name for name in trainable)
    assert count_adapter_parameters(model.accessibility) == sum(
        p.numel() for p in adapter_parameters(model.accessibility)
    )


def test_adapter_bundle_save_load_reproduces_predictions(tmp_path):
    torch.manual_seed(0)
    model = TinyChromBPNet()
    base_state = {
        key: value.detach().clone()
        for key, value in model.accessibility.state_dict().items()
    }
    config = inject_adapters(
        model.accessibility,
        targets=["heads_only"],
        conv_rank=2,
        linear_rank=1,
        conv_scale=0.5,
        linear_scale=1.0,
    )
    X = torch.randn(2, 4, 11)
    for parameter in adapter_parameters(model.accessibility):
        torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    expected = model.accessibility(X)

    manifest = {
        "format_version": 1,
        "adapter_config": {
            "targets": config["targets"],
            "conv_rank": config["conv_rank"],
            "linear_rank": config["linear_rank"],
            "conv_scale": config["conv_scale"],
            "linear_scale": config["linear_scale"],
            "dropout": config["dropout"],
        },
        "layer_specs": config["layer_specs"],
    }
    path = tmp_path / "adapter.pt"
    save_adapter_bundle(path, model, manifest)

    base = TinyChromBPNet()
    base.accessibility.load_state_dict(base_state)

    loaded, _ = load_adapter_bundle(base, path)
    observed = loaded.accessibility(X)

    torch.testing.assert_close(observed[0], expected[0])
    torch.testing.assert_close(observed[1], expected[1])


def test_adapter_manifest_mismatch_fails(tmp_path):
    model = TinyChromBPNet()
    config = inject_adapters(model.accessibility, targets=["linear"], linear_rank=1)
    manifest = {
        "format_version": 1,
        "adapter_config": {
            "targets": config["targets"],
            "conv_rank": config["conv_rank"],
            "linear_rank": config["linear_rank"],
            "conv_scale": config["conv_scale"],
            "linear_scale": config["linear_scale"],
            "dropout": config["dropout"],
        },
        "layer_specs": config["layer_specs"],
    }
    manifest["layer_specs"]["linear"]["out_features"] = 99
    path = tmp_path / "bad_adapter.pt"
    save_adapter_bundle(path, model, manifest)

    with pytest.raises(RuntimeError, match="manifest"):
        load_adapter_bundle(TinyChromBPNet(), path)
