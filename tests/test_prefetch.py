import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing

import pytest

from rl2.prefetch import _End, prefetch


@pytest.mark.parametrize("values", [[], [None, ValueError("data"), _End(), 1]], ids=["empty", "ordered-values"])
def test_prefetch_values(values: list[object]) -> None:
    async def source() -> AsyncGenerator[object]:
        for value in values:
            yield value

    async def run() -> None:
        async with asyncio.timeout(5):
            assert [item async for item in prefetch(source(), n=2)] == values

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 3], ids=["minimum", "multiple"])
def test_prefetch_bounds_and_refills(n: int) -> None:
    async def run() -> None:
        reads: asyncio.Queue[int] = asyncio.Queue()

        async def source() -> AsyncGenerator[int]:
            index = 0
            while True:
                reads.put_nowait(index)
                yield index
                index += 1

        async with asyncio.timeout(5), aclosing(source()) as items, aclosing(prefetch(items, n=n)) as buffered:
            assert await anext(buffered) == 0
            assert [await reads.get() for _ in range(n + 2)] == list(range(n + 2))
            await asyncio.sleep(0)
            assert reads.empty()
            assert await anext(buffered) == 1
            assert await reads.get() == n + 2

        await asyncio.sleep(0)
        assert reads.empty()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        ValueError("sampling failed"),
        asyncio.CancelledError(),
        BaseExceptionGroup("sampling failed", [asyncio.CancelledError()]),
    ],
    ids=["error", "cancelled", "base-exception-group"],
)
def test_prefetch_propagates_error_after_buffered_items(error: BaseException) -> None:
    async def run() -> None:
        failed = asyncio.Event()

        async def source() -> AsyncGenerator[int]:
            yield 1
            yield 2
            failed.set()
            raise error

        async with asyncio.timeout(5), aclosing(prefetch(source(), n=1)) as buffered:
            assert await anext(buffered) == 1
            await failed.wait()
            assert await anext(buffered) == 2
            with pytest.raises(type(error)) as caught:
                await anext(buffered)
            assert caught.value is error

    asyncio.run(run())


def test_prefetch_close_cancels_inflight_read() -> None:
    async def run() -> None:
        reading = asyncio.Event()
        stopped = asyncio.Event()

        async def source() -> AsyncGenerator[int]:
            try:
                yield 1
                yield 2
                reading.set()
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with asyncio.timeout(5), aclosing(source()) as items:
            async with aclosing(prefetch(items, n=1)) as buffered:
                assert await anext(buffered) == 1
                await reading.wait()
            assert stopped.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("yield_on_cancel", [False, True], ids=["stop", "final-item"])
def test_prefetch_close_when_source_handles_cancellation(yield_on_cancel: bool) -> None:
    async def run() -> None:
        reading = asyncio.Event()
        cancelled = asyncio.Event()

        async def source() -> AsyncGenerator[int]:
            yield 1
            yield 2
            reading.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                if yield_on_cancel:
                    yield 3
                    pytest.fail("Source must not be read again during shutdown")

        async with asyncio.timeout(5), aclosing(source()) as items:
            async with aclosing(prefetch(items, n=1)) as buffered:
                assert await anext(buffered) == 1
                await reading.wait()
            assert cancelled.is_set()

    asyncio.run(run())


def test_prefetch_close_propagates_source_cleanup_error() -> None:
    async def run() -> None:
        reading = asyncio.Event()
        error = RuntimeError("source cleanup failed")

        async def source() -> AsyncGenerator[int]:
            try:
                yield 1
                reading.set()
                await asyncio.Event().wait()
            finally:
                raise error

        async with asyncio.timeout(5), aclosing(source()) as items:
            with pytest.raises(RuntimeError, match="source cleanup failed") as caught:
                async with aclosing(prefetch(items, n=1)) as buffered:
                    assert await anext(buffered) == 1
                    await reading.wait()
            assert caught.value is error

    asyncio.run(run())


def test_prefetch_consumer_cancellation() -> None:
    async def run() -> None:
        reading = asyncio.Event()
        stopped = asyncio.Event()

        async def source() -> AsyncGenerator[int]:
            try:
                reading.set()
                await asyncio.Event().wait()
                yield 1
            finally:
                stopped.set()

        async def consume() -> None:
            async with aclosing(source()) as items, aclosing(prefetch(items, n=1)) as buffered:
                await anext(buffered)

        async with asyncio.timeout(5), asyncio.TaskGroup() as pending:
            consumer = pending.create_task(consume())
            await reading.wait()
            consumer.cancel()
        assert consumer.cancelled()
        assert stopped.is_set()

    asyncio.run(run())


def test_prefetch_cancellation_during_close() -> None:
    async def run() -> None:
        reading = asyncio.Event()
        closing = asyncio.Event()

        async def source() -> AsyncGenerator[int]:
            try:
                yield 1
                reading.set()
                await asyncio.Event().wait()
            finally:
                closing.set()
                await asyncio.Event().wait()

        async def consume() -> None:
            async with aclosing(source()) as items, aclosing(prefetch(items, n=1)) as buffered:
                assert await anext(buffered) == 1
                await reading.wait()

        async with asyncio.timeout(5), asyncio.TaskGroup() as pending:
            consumer = pending.create_task(consume())
            await closing.wait()
            consumer.cancel()
        assert consumer.cancelled()

    asyncio.run(run())


def test_prefetch_rejects_nonpositive_capacity() -> None:
    async def source() -> AsyncGenerator[int]:
        yield 1

    async def run() -> None:
        async with aclosing(source()) as items:
            with pytest.raises(ValueError, match="n must be at least 1"):
                await anext(prefetch(items, n=0))

    asyncio.run(run())
