# Converted GDN-2 checkpoints

`paper-matched-30b/` contains the converted
[LLM-OS-Models2 370M paper-matched checkpoint](https://huggingface.co/LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched/tree/6f3458a5d7979ed3d85622b1e71e8eee25895742).

| File | Contents |
| --- | --- |
| `paper-matched-30b/params.npz` | 380,603,648 float32 parameters (about 1.52 GB); local, Git-ignored artifact |
| `paper-matched-30b/manifest.json` | Architecture, training metadata, pinned source revision, and SHA256 checksums |
| `paper-matched-30b/verification.json` | Exact tensor comparison and short-input JAX/PyTorch CPU logit comparison |

Load the model without initializing random parameters or installing PyTorch:

```python
from pathlib import Path
from rl2.gdn2.checkpoints import load_checkpoint

model, variables = load_checkpoint(Path("checkpoints/gdn2/paper-matched-30b"))
# carry, logits = model.apply(variables, token_ids)  # token_ids: [batch, time]
```

The source is `model.pth` at revision `6f3458a5d7979ed3d85622b1e71e8eee25895742`,
training step 57,220. All 355 tensors preserve the source values exactly after
layout conversion. A four-token float32 CPU check against an independent PyTorch
implementation found maximum absolute logit error `2.3603439331054688e-5` and
matching top predictions at every position. This checks the source equations;
it does not run upstream Triton kernels or establish long-context GPU parity.

See [conversion instructions](../../src/rl2/gdn2/README.md#paper-matched-370m-checkpoint).
To repeat the full-checkpoint verification with CPU PyTorch installed:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python tests/verify_gdn2_checkpoint.py \
  --source /tmp/gdn2-paper-matched.pth \
  --checkpoint checkpoints/gdn2/paper-matched-30b
```

Generate a continuation with:

```bash
uv run --extra gdn2 python -m rl2.gdn2.generate \
  --prompt "The capital of France is" --max-new-tokens 100 --temperature 0
```

The script downloads and caches the checksum-verified TinyLlama tokenizer as
`checkpoints/gdn2/tokenizer.model` on first use. Use
`--tokenizer PATH` to select a local SentencePiece model. The converted artifact
contains model weights only, with no optimizer or recurrent inference state.
