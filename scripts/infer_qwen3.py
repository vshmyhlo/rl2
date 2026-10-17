"""Generate text with Qwen3 in JAX.

uv run python scripts/infer_qwen3.py --prompt 'Hello' --temperature 0
Add --attention-backend cudnn for fused NVIDIA GPU attention, or --help for options.
"""

from rl2.qwen3 import main

if __name__ == "__main__":
    main()
