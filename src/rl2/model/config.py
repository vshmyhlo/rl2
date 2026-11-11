"""Tagged configuration for MAPPO actor-critic models."""

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 128
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MLPEncoderConfig:
    type: Literal["mlp"] = "mlp"
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 128


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GenericModelConfig:
    type: Literal["generic"] = "generic"
    encoder: MLPEncoderConfig = MLPEncoderConfig()
    recurrent: LSTMConfig = LSTMConfig()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LuxModelConfig:
    type: Literal["lux"] = "lux"
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 192
    channels: Annotated[int, Field(gt=0, multiple_of=2, strict=True)] = 64
    num_blocks: Annotated[int, Field(gt=0, strict=True)] = 2
    attention_implementation: Literal["triton", "xla"] = "triton"
    recurrent: LSTMConfig = LSTMConfig(hidden_size=256)
    ship_recurrent: LSTMConfig = LSTMConfig(hidden_size=128)


def _default_model_type(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "generic", **value}
    return value


type ModelConfig = Annotated[
    Annotated[GenericModelConfig | LuxModelConfig, Field(discriminator="type")],
    BeforeValidator(_default_model_type),
]
