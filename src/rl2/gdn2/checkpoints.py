"""Strict LitGPT/PyTorch -> rl2 GDN-2 checkpoint conversion and NumPy loading.

Only reading .pth files requires PyTorch. Loading converted checkpoints needs
the normal rl2 dependencies; parameters are float32, pickle-free NPZ arrays.
Run ``python -m rl2.gdn2.checkpoints --help`` for the paper-matched converter.
"""

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from rl2.gdn2.model import GatedDeltaNet2Config, GatedDeltaNet2LM, gdn2_370m
from rl2.shape_checker import ShapeChecker

type Parameters = dict[str, Any]
type StateDict = Mapping[str, np.ndarray]

PAPER_MATCHED_SOURCE = {
    "repo_id": "LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched",
    "revision": "6f3458a5d7979ed3d85622b1e71e8eee25895742",
    "filename": "model.pth",
    "sha256": "acac9407c19cb0941d7d0ee364c60e691fdd7a7d0edf99bd9eee0f4269f70824",
}


@dataclass(frozen=True)
class WeightSpec:
    source: str
    target: str
    shape: tuple[int, ...]
    transform: Literal["identity", "linear", "conv"] = "identity"

    @property
    def target_shape(self) -> tuple[int, ...]:
        if self.transform == "linear":
            return self.shape[::-1]
        if self.transform == "conv":
            return self.shape[2], self.shape[0]
        return self.shape


def weight_specs(model: GatedDeltaNet2LM) -> tuple[WeightSpec, ...]:
    """Exhaustive weight mapping; no random model initialization is needed."""
    c = model.config
    if min(model.num_layers, model.intermediate_size, model.vocab_size) <= 0:
        raise ValueError("num_layers, intermediate_size and vocab_size must be positive")
    d, k = c.hidden_size, c.num_heads * c.head_dim
    v, hv, m = c.value_heads * c.value_head_dim, c.value_head_dim, model.intermediate_size
    specs = [
        WeightSpec("transformer.wte.weight", "embedding/embedding", (model.vocab_size, d)),
        WeightSpec("lm_head.weight", "lm_head/kernel", (model.vocab_size, d), "linear"),
        WeightSpec("transformer.ln_f.weight", "backbone/final_norm/scale", (d,)),
    ]
    for i in range(model.num_layers):
        source, target = f"transformer.h.{i}.", f"backbone/mixer_{i}/"
        specs.extend(
            [
                WeightSpec(source + "norm_1.weight", f"backbone/norm_mixer_{i}/scale", (d,)),
                WeightSpec(source + "norm_2.weight", f"backbone/norm_mlp_{i}/scale", (d,)),
                WeightSpec(source + "mlp.swiglu.w1.weight", f"backbone/mlp_gate_{i}/kernel", (m, d), "linear"),
                WeightSpec(source + "mlp.swiglu.w2.weight", f"backbone/mlp_up_{i}/kernel", (m, d), "linear"),
                WeightSpec(source + "mlp.swiglu.w3.weight", f"backbone/mlp_down_{i}/kernel", (d, m), "linear"),
            ]
        )
        source += "attn."
        specs.extend(
            [
                WeightSpec(source + "A_log", target + "A_log", (c.num_heads,)),
                WeightSpec(source + "dt_bias", target + "dt_bias", (k,)),
                WeightSpec(source + "o_norm.weight", target + "o_norm_scale", (hv,)),
                WeightSpec(source + "g_proj.1.bias", target + "g_proj_out/bias", (v,)),
            ]
        )
        for name, destination, shape in (
            ("q_proj", "q_proj", (k, d)),
            ("k_proj", "k_proj", (k, d)),
            ("v_proj", "v_proj", (v, d)),
            ("b_proj", "b_proj", (k, d)),
            ("w_proj", "w_proj", (v, d)),
            ("f_proj.0", "f_proj_in", (hv, d)),
            ("f_proj.1", "f_proj_out", (k, hv)),
            ("g_proj.0", "g_proj_in", (hv, d)),
            ("g_proj.1", "g_proj_out", (v, hv)),
            ("o_proj", "o_proj", (d, v)),
        ):
            specs.append(WeightSpec(source + name + ".weight", target + destination + "/kernel", shape, "linear"))
        if c.use_short_conv:
            for name, width in (("q", k), ("k", k), ("v", v)):
                specs.append(
                    WeightSpec(
                        source + name + "_conv1d.weight",
                        target + name + "_conv_kernel",
                        (width, 1, c.conv_size),
                        "conv",
                    )
                )
                if c.conv_bias:
                    specs.append(WeightSpec(source + name + "_conv1d.bias", target + name + "_conv_bias", (width,)))
    return tuple(specs)


def _check_array(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    names = "ABC"[: len(shape)]
    sc = ShapeChecker(**dict(zip(names, shape, strict=True)))
    try:
        sc.check(array, names, np.float32)
    except AssertionError as error:
        raise ValueError(f"Invalid weight {name}: {error}") from error
    if not np.isfinite(array).all():
        raise ValueError(f"Weight {name} contains non-finite values")


def _check_keys(actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"Checkpoint keys differ: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def convert_litgpt_state_dict(state: StateDict, model: GatedDeltaNet2LM) -> Parameters:
    """Convert a float32 NumPy state dict, rejecting missing/extra/malformed weights.

    Dense matrices are transposed. Depthwise convolution [channels,1,taps]
    becomes [taps,channels], retaining oldest-to-current tap order. Norm scales,
    gate biases, A_log and dt_bias are copied without reinterpretation.
    """
    specs = weight_specs(model)
    _check_keys(set(state), {spec.source for spec in specs})
    converted: dict[str, np.ndarray] = {}
    for spec in specs:
        array = state[spec.source]
        _check_array(array, spec.shape, spec.source)
        if spec.transform == "linear":
            array = array.T
        elif spec.transform == "conv":
            array = array[:, 0, :].T
        converted[spec.target] = np.ascontiguousarray(array)
    return unflatten_dict(converted, sep="/")


def read_pytorch_checkpoint(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read tensor-only state via PyTorch's restricted weights-only unpickler.

    Supports either a bare state dict or a training checkpoint's ``model`` key.
    Uses memory-mapped CPU storage; optimizer tensors are not converted. Never
    falls back to unrestricted pickle loading or executes downloaded model code.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Reading .pth requires PyTorch >= 2.6; install CPU PyTorch or use uv run --with torch."
        ) from error
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a PyTorch state dict or a checkpoint with a 'model' mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint model must be a nonempty tensor mapping")
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Expected a named floating-point tensor, got {name!r}")
        arrays[name] = tensor.detach().float().numpy()
    metadata: dict[str, Any] = {}
    if "model" in checkpoint:
        for name in ("iter_num", "step_count", "hparams"):
            if name in checkpoint:
                # These are untyped external metadata, not executable state.
                try:
                    metadata[name] = json.loads(json.dumps(checkpoint[name], allow_nan=False))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Checkpoint metadata {name!r} is not JSON serializable") from error
    return arrays, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_config(model: GatedDeltaNet2LM) -> dict[str, Any]:
    mixer = asdict(model.config)
    mixer.pop("dtype")
    return {
        "mixer": mixer,
        "num_layers": model.num_layers,
        "intermediate_size": model.intermediate_size,
        "vocab_size": model.vocab_size,
    }


def _model_from_config(config: Any, dtype: jax.typing.DTypeLike) -> GatedDeltaNet2LM:
    """Validate JSON at the file boundary before constructing typed config."""
    expected = {"mixer", "num_layers", "intermediate_size", "vocab_size"}
    if not isinstance(config, dict) or set(config) != expected or not isinstance(config["mixer"], dict):
        raise ValueError("Invalid checkpoint model configuration")
    for key in expected - {"mixer"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    mixer = config["mixer"]
    expected_mixer = set(asdict(GatedDeltaNet2Config())) - {"dtype"}
    if set(mixer) != expected_mixer:
        raise ValueError("Invalid checkpoint mixer configuration keys")
    for key in ("hidden_size", "head_dim", "num_heads", "conv_size"):
        if type(mixer[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if mixer["num_v_heads"] is not None and type(mixer["num_v_heads"]) is not int:
        raise ValueError("num_v_heads must be an integer or null")
    for key in ("use_short_conv", "conv_bias", "allow_neg_eigval"):
        if type(mixer[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("expand_v", "norm_eps"):
        if type(mixer[key]) not in (int, float):
            raise ValueError(f"{key} must be numeric")
    return GatedDeltaNet2LM(
        GatedDeltaNet2Config(**mixer, dtype=dtype),
        config["num_layers"],
        config["intermediate_size"],
        config["vocab_size"],
    )


def save_checkpoint(
    directory: Path,
    model: GatedDeltaNet2LM,
    params: Parameters,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically create a new checkpoint directory; never overwrite an existing one."""
    if directory.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {directory}")
    flat = flatten_dict(params, sep="/")
    specs = weight_specs(model)
    _check_keys(set(flat), {spec.target for spec in specs})
    for spec in specs:
        _check_array(flat[spec.target], spec.target_shape, spec.target)
    manifest = {
        "format": "rl2.gdn2.npz",
        "version": 1,
        "model": _model_config(model),
        "parameter_count": sum(array.size for array in flat.values()),
        "metadata": dict(metadata),
    }
    json.dumps(manifest, allow_nan=False)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as tmp:
        staging = Path(tmp) / "checkpoint"
        staging.mkdir()
        np.savez(staging / "params.npz", **flat)
        manifest["params_sha256"] = sha256_file(staging / "params.npz")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(directory)


def load_checkpoint(
    directory: Path,
    *,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> tuple[GatedDeltaNet2LM, Parameters]:
    """Load a model and its Flax variables without initializing random parameters.

    Returns ``(model, {"params": ...})`` with float32 NumPy leaves, accepted by
    ``model.apply`` and JAX transformations. dtype controls model computation.
    Verifies the stored checksum, exact parameter names, shapes, and dtypes.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "rl2.gdn2.npz" or manifest.get("version") != 1:
        raise ValueError("Unsupported GDN-2 checkpoint format/version")
    model = _model_from_config(manifest.get("model"), dtype)
    if sha256_file(directory / "params.npz") != manifest.get("params_sha256"):
        raise ValueError("Checkpoint params.npz checksum mismatch")
    specs = weight_specs(model)
    with np.load(directory / "params.npz", allow_pickle=False) as archive:
        _check_keys(set(archive.files), {spec.target for spec in specs})
        flat = {spec.target: archive[spec.target] for spec in specs}
    for spec in specs:
        _check_array(flat[spec.target], spec.target_shape, spec.target)
    if sum(array.size for array in flat.values()) != manifest.get("parameter_count"):
        raise ValueError("Checkpoint parameter count mismatch")
    return model, {"params": unflatten_dict(flat, sep="/")}


def convert_paper_matched_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    """Verify the pinned 30B model.pth and write the local 370M variant."""
    if destination.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {destination}")
    digest = sha256_file(source)
    if digest != PAPER_MATCHED_SOURCE["sha256"]:
        raise ValueError("Source SHA256 does not match the pinned paper-matched model.pth")
    state, training_metadata = read_pytorch_checkpoint(source)
    model = gdn2_370m()
    params = convert_litgpt_state_dict(state, model)
    metadata = {
        "source": PAPER_MATCHED_SOURCE,
        "training": training_metadata,
        "conversion": "Float32 weights; dense transpose and depthwise convolution axis permutation only.",
    }
    save_checkpoint(destination, model, params, metadata)
    return json.loads((destination / "manifest.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Downloaded paper-matched model.pth")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/gdn2/paper-matched-30b"))
    args = parser.parse_args()
    manifest = convert_paper_matched_checkpoint(args.input, args.output)
    print(f"Wrote {manifest['parameter_count']:,} parameters to {args.output}")
    print(f"params.npz SHA256: {manifest['params_sha256']}")


if __name__ == "__main__":
    main()
