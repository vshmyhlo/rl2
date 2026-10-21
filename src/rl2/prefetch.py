import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import cast


@dataclass
class _End:
    error: BaseException | None = None


async def prefetch[T](source: AsyncIterator[T], n: int) -> AsyncGenerator[T]:
    """Buffer up to n items plus one being read or waiting to enqueue, preserving order.

    The item currently yielded to the consumer does not count toward n.
    Use contextlib.aclosing on the result when iteration may stop early.
    The caller retains ownership of the source and its cleanup.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    queue: asyncio.Queue[T | _End] = asyncio.Queue(maxsize=n)
    end = _End()
    closing = False

    async def produce() -> None:
        try:
            while True:
                item = await anext(source)
                # The source may handle cancellation by returning one last item.
                if closing:
                    return
                await queue.put(item)
        except StopAsyncIteration:
            pass
        except BaseException as error:
            if closing:
                raise
            # Deliver failures on consumption, without interrupting the consumer's work.
            end.error = error
        if not closing:
            await queue.put(end)

    producer = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is end:
                if end.error is not None:
                    raise end.error
                return
            yield cast(T, item)
    finally:
        closing = True
        producer.cancel()
        # Ignore producer cancellation while preserving cancellation of the consumer.
        await asyncio.gather(producer, return_exceptions=True)
        if not producer.cancelled():
            producer.result()
