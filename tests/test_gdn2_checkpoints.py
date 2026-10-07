from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2LM, gdn2_370m
from rl2.gdn2.checkpoints import (
    convert_litgpt_state_dict,
    convert_paper_matched_checkpoint,
    load_checkpoint,
    read_pytorch_checkpoint,
    save_checkpoint,
    weight_specs,
)
from rl2.shape_checker import ShapeChecker

if TYPE_CHECKING:
    import torch


@pytest.fixture(scope="module")
def tiny_model() -> GatedDeltaNet2LM:
    return GatedDeltaNet2LM(
        GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=2, conv_size=3),
        2,
        6,
        7,
    )


@pytest.fixture(scope="module")
def source_state() -> dict[str, np.ndarray]:
    """Independent source-layout fixture; do not derive keys from converter specs."""
    rng = np.random.default_rng(17)

    def array(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.normal(size=shape) * 0.15).astype(np.float32)

    result = {
        "transformer.wte.weight": array((7, 4)),
        "lm_head.weight": array((7, 4)),
        "transformer.ln_f.weight": 1 + array((4,)),
    }
    for i in range(2):
        prefix = f"transformer.h.{i}."
        result[prefix + "norm_1.weight"] = 1 + array((4,))
        result[prefix + "norm_2.weight"] = 1 + array((4,))
        for name, shape in (("w1", (6, 4)), ("w2", (6, 4)), ("w3", (4, 6))):
            result[prefix + f"mlp.swiglu.{name}.weight"] = array(shape)
        prefix += "attn."
        result[prefix + "A_log"] = np.log(np.array([2, 3], np.float32))
        result[prefix + "dt_bias"] = -3 + array((4,))
        result[prefix + "o_norm.weight"] = 1 + array((2,))
        result[prefix + "g_proj.1.bias"] = array((4,))
        for name in ("q_proj", "k_proj", "v_proj", "b_proj", "w_proj", "o_proj"):
            result[prefix + name + ".weight"] = array((4, 4))
        for name in ("f_proj", "g_proj"):
            result[prefix + name + ".0.weight"] = array((2, 4))
            result[prefix + name + ".1.weight"] = array((4, 2))
        for name in ("q", "k", "v"):
            result[prefix + name + "_conv1d.weight"] = array((4, 1, 3))
    return result


def torch_reference_logits(
    state: dict[str, np.ndarray],
    model: GatedDeltaNet2LM,
    tokens: np.ndarray,
) -> np.ndarray:
    """CPU reproduction of source LitGPT equations, without the converter mapping.

    This also supports manually verifying actual downloaded checkpoints. It uses
    PyTorch linear/conv/norm operations and explicit dense transition matrices.
    It does not import or execute any downloaded model code or Triton kernels.
    """
    import torch
    from torch.nn import functional as f

    sc = ShapeChecker()
    sc.check(tokens, "TB", np.int32)
    c = model.config
    weights = {key: torch.from_numpy(value) for key, value in state.items()}

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        return f.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x: torch.Tensor, name: str) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.norm_eps) * weights[name + ".weight"]

    with torch.no_grad():
        x = f.embedding(torch.from_numpy(tokens.astype(np.int64)), weights["transformer.wte.weight"])
        for i in range(model.num_layers):
            layer = f"transformer.h.{i}."
            attn = layer + "attn."
            a = norm(x, layer + "norm_1")
            qkv = []
            for name in ("q", "k", "v"):
                projected = linear(a, attn + name + "_proj")
                if c.use_short_conv:
                    projected = f.conv1d(
                        projected.permute(1, 2, 0),
                        weights[attn + name + "_conv1d.weight"],
                        weights.get(attn + name + "_conv1d.bias"),
                        padding=c.conv_size - 1,
                        groups=projected.shape[-1],
                    )[..., : len(tokens)].permute(2, 0, 1)
                qkv.append(f.silu(projected))
            q, k, v = qkv
            key_shape = (*tokens.shape, c.num_heads, c.head_dim)
            value_shape = (*tokens.shape, c.value_heads, c.value_head_dim)
            q, k = q.reshape(key_shape), k.reshape(key_shape)
            q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * c.head_dim**-0.5
            k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            decay = linear(linear(a, attn + "f_proj.0"), attn + "f_proj.1")
            decay = -weights[attn + "A_log"].exp().repeat_interleave(c.head_dim) * f.softplus(
                decay + weights[attn + "dt_bias"]
            )
            erase = torch.sigmoid(linear(a, attn + "b_proj"))
            if c.allow_neg_eigval:
                erase = 2 * erase
            write = torch.sigmoid(linear(a, attn + "w_proj")).reshape(value_shape)
            q, k, decay, erase = (
                t.reshape(key_shape).repeat_interleave(c.value_heads // c.num_heads, dim=2)
                for t in (q, k, decay, erase)
            )
            v = v.reshape(value_shape)
            memory = torch.zeros((tokens.shape[1], c.value_heads, c.head_dim, c.value_head_dim))
            identity = torch.eye(c.head_dim)
            outputs = []
            for t in range(len(tokens)):
                transition = identity - k[t][..., :, None] * (erase[t] * k[t])[..., None, :]
                memory = transition @ (decay[t].exp()[..., None] * memory)
                memory = memory + k[t][..., :, None] * (write[t] * v[t])[..., None, :]
                outputs.append((q[t][..., None, :] @ memory).squeeze(-2))
            y = norm(torch.stack(outputs), attn + "o_norm")
            gate = linear(linear(a, attn + "g_proj.0"), attn + "g_proj.1").reshape(value_shape)
            x = x + linear((y * f.silu(gate)).flatten(-2), attn + "o_proj")
            a = norm(x, layer + "norm_2")
            hidden = f.silu(linear(a, layer + "mlp.swiglu.w1")) * linear(a, layer + "mlp.swiglu.w2")
            x = x + linear(hidden, layer + "mlp.swiglu.w3")
        output = linear(norm(x, "transformer.ln_f"), "lm_head").numpy()
    sc = ShapeChecker(V=model.vocab_size)
    sc.check(tokens, "TB", np.int32)
    sc.check(output, "TBV", np.float32)
    return output


def test_370m_preset_matches_checkpoint_dimensions() -> None:
    model = gdn2_370m()
    assert (model.config.hidden_size, model.config.num_heads, model.config.head_dim) == (1024, 16, 128)
    assert (model.num_layers, model.intermediate_size, model.vocab_size) == (16, 2048, 32000)
    assert sum(np.prod(spec.shape) for spec in weight_specs(model)) == 380_603_648


def test_conversion_roundtrip_and_logits(
    tmp_path: Path,
    tiny_model: GatedDeltaNet2LM,
    source_state: dict[str, np.ndarray],
) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    params = convert_litgpt_state_dict(source_state, tiny_model)
    np.testing.assert_array_equal(params["lm_head"]["kernel"], source_state["lm_head.weight"].T)
    np.testing.assert_array_equal(
        params["backbone"]["mixer_0"]["q_conv_kernel"], source_state["transformer.h.0.attn.q_conv1d.weight"][:, 0, :].T
    )
    tokens = np.array([[1], [2], [3]], np.int32)
    expected = torch_reference_logits(source_state, tiny_model, tokens)
    destination = tmp_path / "model"
    save_checkpoint(destination, tiny_model, params, {"step_count": 12})
    loaded_model, variables = load_checkpoint(destination)
    for name, array in flatten_dict(variables["params"], sep="/").items():
        np.testing.assert_array_equal(array, flatten_dict(params, sep="/")[name])
    carry, actual = jax.jit(loaded_model.apply)(variables, jnp.asarray(tokens))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=3e-5)
    assert len(carry) == 2
    bf16_model, _ = load_checkpoint(destination, dtype=jnp.bfloat16, backend="triton")
    assert bf16_model.backend == "triton"
    assert bf16_model.config.dtype == jnp.bfloat16
    assert json.loads((destination / "manifest.json").read_text())["metadata"] == {"step_count": 12}
    with pytest.raises(FileExistsError):
        save_checkpoint(destination, tiny_model, params, {})


@pytest.mark.parametrize("wrapped", [False, True], ids=["bare-state", "training-checkpoint"])
def test_safe_pytorch_reader(
    tmp_path: Path,
    source_state: dict[str, np.ndarray],
    wrapped: bool,
) -> None:
    torch = pytest.importorskip("torch")
    state = {k: torch.from_numpy(v.copy()) for k, v in source_state.items()}
    checkpoint = {"model": state, "optimizer": None, "step_count": 12} if wrapped else state
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)
    arrays, metadata = read_pytorch_checkpoint(path)
    for key in state:
        np.testing.assert_array_equal(arrays[key], source_state[key])
    assert metadata == ({"step_count": 12} if wrapped else {})
    torch.save({"model": {"not_a_tensor": "invalid"}}, path)
    with pytest.raises(ValueError, match="floating-point tensor"):
        read_pytorch_checkpoint(path)


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "dtype", "nonfinite"])
def test_invalid_source_weights(
    tiny_model: GatedDeltaNet2LM,
    source_state: dict[str, np.ndarray],
    problem: str,
) -> None:
    state = source_state.copy()
    name = "transformer.h.0.attn.q_conv1d.weight"
    if problem == "missing":
        state.pop(name)
    elif problem == "unexpected":
        state["unused.weight"] = np.zeros((1,), np.float32)
    elif problem == "shape":
        state[name] = state[name][:, 0]
    elif problem == "dtype":
        state[name] = state[name].astype(np.int32)
    else:
        state[name] = np.full_like(state[name], np.nan)
    with pytest.raises(ValueError):
        convert_litgpt_state_dict(state, tiny_model)


def test_checkpoint_integrity_and_manifest_validation(
    tmp_path: Path,
    tiny_model: GatedDeltaNet2LM,
    source_state: dict[str, np.ndarray],
) -> None:
    params = convert_litgpt_state_dict(source_state, tiny_model)
    directory = tmp_path / "model"
    save_checkpoint(directory, tiny_model, params, {})
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    invalid = dict(manifest, version=99)
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="format/version"):
        load_checkpoint(directory)
    invalid = dict(manifest, model={**manifest["model"], "num_layers": "2"})
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="positive integer"):
        load_checkpoint(directory)
    manifest_path.write_text(json.dumps(manifest))
    with (directory / "params.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(directory)


def test_pinned_source_rejects_wrong_file(tmp_path: Path) -> None:
    source = tmp_path / "wrong.pth"
    source.write_bytes(b"not the pinned checkpoint")
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="SHA256"):
        convert_paper_matched_checkpoint(source, destination)
    assert not destination.exists()


def test_cli_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from rl2.gdn2 import checkpoints

    source, target = tmp_path / "input.pth", tmp_path / "output"

    def convert(input_path: Path, output_path: Path) -> dict[str, Any]:
        assert (input_path, output_path) == (source, target)
        return {"parameter_count": 123, "params_sha256": "abc"}

    monkeypatch.setattr(checkpoints, "convert_paper_matched_checkpoint", convert)
    monkeypatch.setattr("sys.argv", ["checkpoints", "--input", str(source), "--output", str(target)])
    checkpoints.main()
    assert "Wrote 123 parameters" in capsys.readouterr().out
