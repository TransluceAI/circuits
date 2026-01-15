"""
Perplexity API client for web search functionality.
"""

import asyncio
from typing import cast

import backoff
from backoff.types import Details
from env_util import ENV
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tqdm.asyncio import tqdm_asyncio
from util.types import ChatMessage


def get_perplexity_client_async() -> AsyncOpenAI:
    """Get an asynchronous Perplexity API client."""
    return AsyncOpenAI(api_key=ENV.PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")


def _print_backoff_message(e: Details):
    print(
        f"Perplexity API backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_pplx_completion_async(
    client: AsyncOpenAI,
    messages: list[ChatMessage],
    model: str = "sonar-pro",
    temperature: float = 1.0,
    timeout: float = 30.0,
) -> ChatCompletion:
    async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():  # type: ignore
        return await client.chat.completions.create(
            model=model,
            messages=[cast(ChatCompletionMessageParam, m) for m in messages],
            temperature=temperature,
        )


async def get_pplx_completions_async(
    client: AsyncOpenAI,
    messages_list: list[list[ChatMessage]],
    model: str = "sonar-pro",
    temperature: float = 1.0,
    max_concurrency: int = 100,
    timeout: float = 30.0,
    use_tqdm: bool = False,
):
    """
    Get completions for multiple message sequences in parallel using Perplexity API.

    Args:
        client (AsyncOpenAI): The async Perplexity API client
        messages_list (list[list[ChatMessage]]): List of message sequences to process
        model (str): Model name to use
        temperature (float): Sampling temperature
        max_concurrency (int): Maximum number of concurrent requests
        timeout (float): Timeout in seconds for each request
        use_tqdm (bool): Whether to show progress bar

    Returns:
        list[ChatCompletion]: List of completion responses in the same order as input messages
    """
    # Create a semaphore to limit concurrency
    sem = asyncio.Semaphore(max_concurrency)

    async def process_one_messages(messages: list[ChatMessage]) -> ChatCompletion:
        async with sem:
            return await _get_pplx_completion_async(
                client=client,
                messages=messages,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )

    # Create tasks for all message sequences
    tasks = [process_one_messages(messages) for messages in messages_list]

    # Use tqdm if requested
    if use_tqdm:
        results = cast(list[ChatCompletion | None], await tqdm_asyncio.gather(*tasks))  # type: ignore
    else:
        results = await asyncio.gather(*tasks)

    return results


def parse_pplx_completion(response: ChatCompletion | None) -> str | None:
    if response is None:
        return None
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError):
        return None
