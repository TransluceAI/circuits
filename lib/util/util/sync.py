import asyncio
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Coroutine, Optional, TypeVar, cast

from tqdm.asyncio import tqdm as tqdm_asyncio

T = TypeVar("T")


def sync_wrapper(coroutine: Coroutine[Any, Any, T]) -> T:
    executor = ThreadPoolExecutor()
    try:
        future: Future[T] = executor.submit(
            asyncio.run,
            coroutine,
        )
        return future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def parallelize_async(
    coroutines: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 100,
    use_tqdm: bool = False,
) -> list[Optional[T]]:
    """
    Parallel execution of coroutines with concurrency control.

    Args:
        coroutines: List of coroutines to execute
        max_concurrency: Maximum number of concurrent tasks
        use_tqdm: Whether to show a progress bar
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(coro: Coroutine[Any, Any, T]) -> Optional[T]:
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                print(f"Task failed: {e.__class__.__name__}")
                traceback.print_exc()
                return None

    tasks = [limited_task(coro) for coro in coroutines]

    if use_tqdm:
        results = cast(
            list[Optional[T]],
            await tqdm_asyncio.gather(*tasks, desc="Processing items"),  # type: ignore
        )
    else:
        results = await asyncio.gather(*tasks)

    return results
