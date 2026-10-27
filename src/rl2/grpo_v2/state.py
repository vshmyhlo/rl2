"""Importable state types preserve Flax pytree registration in Ray workers."""

from flax import struct
from flax.training.train_state import TrainState

from rl2.qwen3 import LoRAParameters, Parameters


class LoRATrainState(TrainState):
    params: LoRAParameters  # Trainable (R, I) and (O, R) leaves.
    base_params: Parameters  # Frozen HF parameter shapes.
    lora_scale: float = struct.field(pytree_node=False)
