"""Register the small checkpoint with BFCL's unchanged Qwen FC handler."""

from dataclasses import replace

from bfcl_eval.__main__ import cli
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

MODEL_CONFIG_MAPPING["Qwen/Qwen3-0.6B-FC"] = replace(
    MODEL_CONFIG_MAPPING["Qwen/Qwen3-8B-FC"],
    model_name="Qwen/Qwen3-0.6B",
    display_name="Qwen3-0.6B (FC)",
    url="https://huggingface.co/Qwen/Qwen3-0.6B",
)

if __name__ == "__main__":
    cli()
