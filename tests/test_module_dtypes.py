"""Compute dtype constructor contracts and inference without forced casts."""

import inspect

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from rl2.alphazero.model import PolicyValueNet
from rl2.ast_transformer import ASTTransformer
from rl2.edit_transformer import EditTransformer
from rl2.gdn2.model import (
    GatedDeltaNet2,
    GatedDeltaNet2Config,
    GatedDeltaNet2LM,
    GatedDeltaNet2Recurrent,
    GatedDeltaNet2Stack,
)
from rl2.gemma3 import Gemma3LM
from rl2.karel_model import KarelProgramModel
from rl2.lstm import LSTM, LSTMStack
from rl2.mamba3 import Mamba3, Mamba3Stack
from rl2.model.encoders.conv2d import ConvObservationEncoder, ConvObservationStage, ConvStage, ResidualBlock
from rl2.model.encoders.craftax import CraftaxEncoder
from rl2.model.encoders.mlp import MLPEncoder
from rl2.model.encoders.xminigrid import XMiniGridEncoder
from rl2.model.generic import GenericActorCritic
from rl2.model.lux import (
    LuxActorCritic,
    LuxAttentionBlock,
    LuxEncoder,
    LuxMapEncoder,
    LuxModel,
    _Attention,
    _GlobalMemory,
    _SetAttentionBlock,
)
from rl2.observation_decoder import ConvObservationDecoder
from rl2.ppo import ActorCritic
from rl2.ppo_rnd import ActorCritic as RNDActorCritic
from rl2.ppo_rnd import ResetLSTM, RNDNetwork
from rl2.ppo_rnd import ResidualBlock as RNDResidualBlock
from rl2.qwen3 import Qwen3Config, Qwen3LM
from rl2.resize_conv import ResizeConv
from rl2.transformer import ARTransformer, BDTransformer, Transformer, _TransformerBlock
from rl2.tree_attention import TreeAttentionBias
from rl2.wm import LSTMWorldModel, MambaWorldModel


@pytest.mark.parametrize(
    "module",
    [
        PolicyValueNet,
        ASTTransformer,
        EditTransformer,
        GatedDeltaNet2,
        GatedDeltaNet2Stack,
        GatedDeltaNet2Recurrent,
        GatedDeltaNet2LM,
        Gemma3LM,
        KarelProgramModel,
        LSTM,
        LSTMStack,
        Mamba3,
        Mamba3Stack,
        ConvObservationEncoder,
        ConvObservationStage,
        ResidualBlock,
        CraftaxEncoder,
        MLPEncoder,
        XMiniGridEncoder,
        GenericActorCritic,
        LuxActorCritic,
        LuxAttentionBlock,
        LuxEncoder,
        LuxMapEncoder,
        LuxModel,
        _Attention,
        _GlobalMemory,
        _SetAttentionBlock,
        ConvObservationDecoder,
        ActorCritic,
        RNDActorCritic,
        ResetLSTM,
        RNDNetwork,
        RNDResidualBlock,
        Qwen3LM,
        ResizeConv,
        ARTransformer,
        BDTransformer,
        Transformer,
        _TransformerBlock,
        TreeAttentionBias,
        LSTMWorldModel,
        MambaWorldModel,
    ],
    ids=lambda module: f"{module.__module__}.{module.__name__}",
)
def test_dtype_is_last_optional_constructor_argument(module: type[nn.Module] | type[Gemma3LM]) -> None:
    parameters = inspect.signature(module).parameters
    # Linen appends its own parent/name arguments after user-defined fields.
    fields = [name for name in parameters if name not in ("parent", "name")]
    assert fields[-1] == "dtype"
    assert parameters["dtype"].default is None


@pytest.mark.parametrize(
    "model,shape,output_shape",
    [
        (MLPEncoder(hidden_size=4, embedding_size=4), (1, 4), (1, 4)),
        (ResidualBlock(channels=2), (1, 2, 2, 2), (1, 2, 2, 2)),
        (RNDResidualBlock(channels=2), (1, 2, 2, 2), (1, 2, 2, 2)),
        (ResizeConv(channels=2, spatial_shape=(2, 2)), (1, 2, 2, 2), (1, 2, 2, 2)),
        (
            ConvObservationDecoder(observation_shape=(1, 2, 2), stages=(ConvStage(channels=2, blocks=0),)),
            (1, 4),
            (1, 1, 2, 2),
        ),
    ],
    ids=["mlp", "residual", "rnd-residual", "resize", "decoder"],
)
def test_none_infers_dtype_from_inputs_and_parameters(
    model: nn.Module, shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> None:
    x = jnp.ones(shape, dtype=jnp.bfloat16)  # (*shape,)
    variables = model.init(jax.random.key(0), x)  # parameter leaves: (...,)
    default = model.apply(variables, x)  # (*output_shape,)
    assert default.shape == output_shape
    assert default.dtype == jnp.float32
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), variables)  # parameter leaves: (...,)
    inferred = jax.jit(model.apply)(variables, x)  # (*output_shape,)
    assert inferred.shape == output_shape
    assert inferred.dtype == jnp.bfloat16


def test_transformer_none_preserves_inferred_output_and_cache_dtype() -> None:
    model = Transformer(dim=4, num_layers=1, num_heads=1, max_seq_len=2, final_norm=False)
    x = jnp.ones((1, 2, 4), dtype=jnp.bfloat16)  # (B, T, D)
    variables = model.init(jax.random.key(0), x)  # parameter leaves: (...,)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), variables)  # parameter leaves: (...,)
    carry, output = jax.jit(model.apply)(variables, x)  # carry: (B, C, H, d); output: (B, T, D)
    assert output.shape == x.shape
    assert output.dtype == jnp.bfloat16
    assert carry[0].key.shape == (1, 2, 1, 4)
    assert carry[0].key.dtype == carry[0].value.dtype == jnp.bfloat16


@pytest.mark.parametrize("kind", ["mixer", "stack", "recurrent", "lm"])
def test_gdn_module_dtype_overrides_config(kind: str) -> None:
    config = GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1, conv_size=2, dtype=jnp.float32)
    if kind == "mixer":
        model = GatedDeltaNet2(config=config, dtype=jnp.bfloat16)
    elif kind == "stack":
        model = GatedDeltaNet2Stack(config=config, num_layers=1, intermediate_size=4, dtype=jnp.bfloat16)
    elif kind == "recurrent":
        model = GatedDeltaNet2Recurrent(config=config, num_layers=1, intermediate_size=4, dtype=jnp.bfloat16)
    else:
        model = GatedDeltaNet2LM(config=config, num_layers=1, intermediate_size=4, vocab_size=4, dtype=jnp.bfloat16)
    x = jnp.ones((1, 1, 4), dtype=jnp.float32)  # (B, T, D), also (T, B, D) for recurrent
    lengths = jnp.ones((1,), dtype=jnp.int32)  # (B,)
    if kind == "recurrent":
        args = (x, model.initial_carry(1), jnp.zeros((1, 1), dtype=jnp.bool_))
    elif kind == "lm":
        args = (jnp.ones((1, 1), dtype=jnp.int32), lengths)
    else:
        args = (x, lengths)
    (carry, output), variables = jax.eval_shape(
        lambda key: model.init_with_output(key, *args, capture_intermediates=True), jax.random.key(0)
    )  # carry leaves: (...,); output: (B, T, D/V)
    assert output.shape == (1, 1, 4)
    assert output.dtype == (jnp.float32 if kind == "lm" else jnp.bfloat16)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(carry))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables["params"]))
    if kind == "lm":
        hidden = variables["intermediates"]["backbone"]["__call__"][0][1]  # (B, T, D)
        assert hidden.shape == (1, 1, 4)
        assert hidden.dtype == jnp.bfloat16


@pytest.mark.parametrize("dtype,param_dtype", [(None, jnp.float32), (None, jnp.bfloat16), (jnp.bfloat16, jnp.float32)])
def test_qwen_optional_dtype_and_cache(dtype: jax.typing.DTypeLike | None, param_dtype: jax.typing.DTypeLike) -> None:
    model = Qwen3LM(
        config=Qwen3Config(
            vocab_size=4,
            hidden_size=4,
            intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=2,
            eos_token_id=1,
        ),
        param_dtype=param_dtype,
        attention_backend="xla",
        dtype=dtype,
    )
    tokens = jnp.ones((1, 1), dtype=jnp.int32)  # (B, T)
    cache = model.init_cache(batch_size=1, capacity=2)  # key/value: (L, B, C, H, d)
    output, variables = jax.eval_shape(lambda key: model.init_with_output(key, tokens, cache=cache), jax.random.key(0))
    assert output.logits.shape == (1, 1, 4)
    assert output.logits.dtype == jnp.float32
    assert output.cache.key.shape == (1, 1, 2, 1, 4)
    assert output.cache.key.dtype == (param_dtype if dtype is None else dtype)
    assert all(leaf.dtype == param_dtype for leaf in jax.tree.leaves(variables["params"]))
