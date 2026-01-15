import asyncio
import traceback
from contextlib import nullcontext
from functools import partial
from typing import cast

from env_util import ENV
from together import AsyncTogether
from together.types import ChatCompletionResponse
from tqdm.asyncio import tqdm_asyncio
from util.types import ChatMessage


def get_together_client_async():
    if ENV.TOGETHER_API_KEY is None:
        raise Exception("TOGETHER_API_KEY is not set; check your .env")
    return AsyncTogether(
        api_key=ENV.TOGETHER_API_KEY,
    )


def bare_message(message: ChatMessage) -> dict[str, str]:
    return {
        "role": message["role"],
        "content": message["content"],
    }


async def _get_together_chat_completion_async(
    client: AsyncTogether,
    messages: list[ChatMessage],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 5.0,
):
    async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():  # type: ignore
        output = await client.chat.completions.create(
            model=model_name,
            messages=[bare_message(message) for message in messages],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        assert isinstance(output, ChatCompletionResponse)
        return output


async def get_together_chat_completions_async(
    client: AsyncTogether,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    max_concurrency: int | None = 100,
    timeout: float = 5.0,
    use_tqdm: bool = False,
    semaphore: asyncio.Semaphore | None = None,
):
    """
    If max_concurrency is provided, it will be used to limit the number of concurrent requests.
    If not, the semaphore will be used to limit the number of concurrent requests.
    If neither is provided, no concurrency control will be applied.
    """
    base_func = partial(
        _get_together_chat_completion_async,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )

    if max_concurrency is not None:
        cm = asyncio.Semaphore(max_concurrency)
    else:
        cm = nullcontext() if semaphore is None else semaphore

    async def limited_task(messages: list[ChatMessage]):
        async with cm:
            try:
                return await base_func(client=client, messages=messages)
            except Exception as e:
                print(
                    f"OpenAI chat completion failed even with backoff: {e.__class__.__name__}\n"
                    f"Failure traceback:\n{traceback.format_exc()}\n"
                    f"Returning None."
                )
                return None

    tasks = [limited_task(messages) for messages in messages_list]
    if use_tqdm:
        responses = cast(
            list[ChatCompletionResponse | None],
            await tqdm_asyncio.gather(*tasks, desc="Processing messages"),  # type: ignore
        )
    else:
        responses = await asyncio.gather(*tasks)

    return responses


def parse_together_completion(response: ChatCompletionResponse) -> str | None:
    if response.choices is None or response.choices[0].message is None:
        return None
    assert isinstance(response.choices[0].message.content, str), "Unexpected message content type"
    return response.choices[0].message.content
