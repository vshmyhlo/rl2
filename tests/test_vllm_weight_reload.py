"""Check the tied-head reload path used by full-weight GRPO publication."""

from unittest.mock import Mock

import pytest
import torch
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


def test_tied_head_logits_follow_embedding_only_reloads(caplog: pytest.LogCaptureFixture) -> None:
    with (
        torch.device("cpu"),
        torch.no_grad(),
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
    ):
        embedding = VocabParallelEmbedding(
            num_embeddings=5, embedding_dim=4, params_dtype=torch.bfloat16, disable_tp=True
        )
        head = ParallelLMHead(num_embeddings=5, embedding_dim=4, params_dtype=torch.bfloat16, disable_tp=True)
        head.tie_weights(embedding)
        model = torch.nn.Module()
        model.add_module("model", torch.nn.Module())
        model.model.add_module("embed_tokens", embedding)
        model.add_module("lm_head", head)
        processor = LogitsProcessor(vocab_size=5)
        processor.head_dtype = torch.float32
        # No attention layers: finalization does not consult the model config.
        config = Mock(spec=ModelConfig)
        record_metadata_for_reloading(model)
        weights = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) / 8  # (V=5, D=4)
        AutoWeightsLoader(model).load_weights([("model.embed_tokens.weight", weights)])
        shared_parameter = embedding.weight  # (P=64, D), including vocabulary padding.
        storage = shared_parameter.data_ptr()
        hidden = torch.tensor([[1, -2, 3, 1], [-1, 1, 2, -3]], dtype=torch.bfloat16)  # (B=2, D)
        previous = processor(head, hidden, skip_gather=True)[:, :5]  # (B, V), discard vocabulary padding.

        for version in (1, 2):
            updated = weights + version * torch.arange(5, dtype=torch.bfloat16)[:, None]  # (V, D)
            initialize_layerwise_reload(model)
            # The learner publishes the shared embedding once, without lm_head.weight.
            AutoWeightsLoader(model).load_weights([("model.embed_tokens.weight", updated)])
            finalize_layerwise_reload(model, config)

            assert embedding.weight is head.weight is shared_parameter
            assert head.weight.data_ptr() == storage
            torch.testing.assert_close(head.weight[:5], updated, rtol=0, atol=0)  # (V, D)
            torch.testing.assert_close(head.weight[5:], torch.zeros_like(head.weight[5:]), rtol=0, atol=0)
            padded_logits = processor(head, hidden, skip_gather=True)  # (B, P)
            assert padded_logits.shape == (2, 64)
            logits = padded_logits[:, :5]  # (B, V)
            expected = hidden.float() @ updated.float().T  # (B, V)
            assert logits.shape == (2, 5) and logits.dtype == torch.float32
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)
            assert not torch.equal(logits, previous)
            previous = logits  # (B, V)

        assert "ParallelLMHead: Failed to load weights" in caplog.text
