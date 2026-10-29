import asyncio
import logging
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI, _base_client

from rl2.vllm_support.retry_logging import add_retry_cause


@pytest.mark.parametrize(
    "failure,recover,cause",
    [
        (None, True, None),
        (503, True, "HTTPStatusError status=503"),
        (httpx.ReadError, True, "ReadError"),
        (httpx.ReadTimeout, True, "ReadTimeout"),
        (400, False, None),
        pytest.param(503, False, "HTTPStatusError status=503", id="exhausted-retries"),
    ],
)
def test_sdk_retry_cause(
    failure: int | type[httpx.RequestError] | None,
    recover: bool,
    cause: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("openai._base_client")
    monkeypatch.setattr(log, "filters", [])
    log.addFilter(add_retry_cause)
    caplog.set_level(logging.INFO, logger=log.name)
    monkeypatch.setattr(_base_client.anyio, "sleep", AsyncMock())
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)  # Exercise exception context across asynchronous calls.
        if failure is not None and (len(requests) == 1 or not recover):
            if isinstance(failure, int):
                return httpx.Response(status_code=failure, json={"error": {"message": "private-error-body"}})
            raise failure("private-exception-details", request=request)
        return httpx.Response(
            status_code=200,
            json={
                "id": "completion",
                "object": "text_completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "text": "done", "finish_reason": "stop"}],
            },
        )

    async def run() -> None:
        async with AsyncOpenAI(
            base_url="http://vllm/v1",
            api_key="private-key",
            max_retries=1,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            response = await client.completions.create(model="test", prompt="private-prompt", max_tokens=1)
            assert response.choices[0].text == "done"

    if recover:
        asyncio.run(run())
    else:
        with pytest.raises(APIStatusError) as caught:
            asyncio.run(run())
        assert caught.value.status_code == failure
    retry_messages = [record.getMessage() for record in caplog.records if record.name == log.name]
    if cause is None:
        assert retry_messages == []
        assert len(requests) == 1
    else:
        assert len(retry_messages) == 1 and len(requests) == 2
        assert retry_messages[0].startswith("Retrying request to /completions in ")
        assert retry_messages[0].endswith(f"(cause={cause})")
    assert "private-" not in caplog.text
