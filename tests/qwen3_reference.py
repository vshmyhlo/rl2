"""Regenerate the tiny HF parity fixture with transformers==4.57.1 and CPU torch."""

from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    rng = np.random.default_rng(7)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = rng.normal(0, 0.2, parameter.shape).astype(np.float32)  # Parameter shape.
            if "norm" in name:
                value += 1
            parameter.copy_(torch.from_numpy(value))
    fixture = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    tokens = torch.tensor([[1, 4, 7], [3, 2, 6]])  # (B=2, T=3)
    positions = torch.tensor([[0, 1, 2], [9, 10, 11]])  # (B, T)
    fixture["tokens"] = tokens.numpy()  # (B, T)
    fixture["positions"] = positions.numpy()  # (B, T)
    fixture["untied_logits"] = model(tokens, position_ids=positions).logits.detach().numpy()  # (B, T, V=13)
    model.lm_head.weight = model.model.embed_tokens.weight
    logits = model(tokens, position_ids=positions).logits  # (B, T, V)
    fixture["tied_logits"] = logits.detach().numpy().copy()  # (B, T, V)
    logits.square().mean().backward()
    fixture["embedding_grad"] = model.model.embed_tokens.weight.grad.numpy().copy()  # (V, D=8)
    destination = Path(__file__).with_name("data") / "qwen3_reference.npz"
    destination.parent.mkdir(exist_ok=True)
    np.savez(destination, **fixture)
    print(destination)


if __name__ == "__main__":
    main()
