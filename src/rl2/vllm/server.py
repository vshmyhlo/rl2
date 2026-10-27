"""vLLM server with one-shot adapter reloads for the pinned vLLM 0.32 release."""

import runpy

from vllm.entrypoints.openai.models import serving
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest


class ReloadOnceServingModels(serving.OpenAIServingModels):
    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        response = await super().load_lora_adapter(request, base_model_name=base_model_name)
        if isinstance(response, str):
            # vLLM 0.32 otherwise reloads these files on every decoding step.
            self.lora_requests[request.lora_name].load_inplace = False
        return response


def main() -> None:
    serving.OpenAIServingModels = ReloadOnceServingModels
    runpy.run_module("vllm.entrypoints.launchers.api_server.entry", run_name="__main__")


if __name__ == "__main__":
    main()
