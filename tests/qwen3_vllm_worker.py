"""Read-only weight inspection for the opt-in LoRA publication test."""

import hashlib
from typing import Any

import torch


class ParityWorker:
    def weight_digests(self: Any) -> dict[str, str]:
        return {
            name.replace(".base_layer.", "."): hashlib.sha256(
                weight.detach().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
            for name, weight in self.get_model().named_parameters()  # Packed vLLM parameter shapes.
        }
