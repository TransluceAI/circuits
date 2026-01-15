import asyncio
import traceback
from contextlib import nullcontext
from functools import partial
from typing import cast

import backoff
from anthropic import Anthropic, AsyncAnthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message, MessageParam
from backoff.types import Details
from env_util import ENV
from tqdm.asyncio import tqdm_asyncio
from util.types import ChatMessage


def _print_backoff_message(e: Details):
    print(
        f"Anthropic backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_anthropic_chat_completion_async(
    client: AsyncAnthropic,
    messages: list[ChatMessage],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    timeout: float = 5.0,
) -> Message:
    """
    Note from kevin 1/29/2025:
        logprobs and top_logprobs were recently added to the OpenAI endpoint,
        which broke some of my code. I'm just adding it to Anthropic as well, to maintain
        "compatibility".

        We should actually implement this at some point, but it does not work.
    """

    if logprobs or top_logprobs is not None:
        raise ValueError(
            "logprobs and top_logprobs are not supported by Anthropic's chat completion endpoint."
        )

    system = messages[0]["content"] if messages[0]["role"] == "system" else None

    async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():  # type: ignore
        return await client.messages.create(
            model=model_name,
            max_tokens=max_new_tokens,
            messages=[
                MessageParam(role=m["role"], content=m["content"])
                for m in messages
                if m["role"] != "system"
            ],
            temperature=temperature,
            system=system if system is not None else NOT_GIVEN,
        )


def get_anthropic_client_sync() -> Anthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return Anthropic(api_key=ENV.ANTHROPIC_API_KEY)


def get_anthropic_client_async() -> AsyncAnthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return AsyncAnthropic(api_key=ENV.ANTHROPIC_API_KEY)


async def get_anthropic_chat_completions_async(
    client: AsyncAnthropic,
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
    print("WARN: util/anthropic.py is deprecated; migrate to util/llm/prod_llms.py instead.")

    base_func = partial(
        _get_anthropic_chat_completion_async,
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
            except Exception as e:  # This will catch any exception, including those from backoff
                print(
                    f"Anthropic chat completion failed even with backoff: {e.__class__.__name__}\n"
                    f"Failure traceback:\n{traceback.format_exc()}\n"
                    f"Returning None."
                )
                return None

    tasks = [limited_task(messages) for messages in messages_list]
    if use_tqdm:
        responses = cast(
            list[Message | None],
            await tqdm_asyncio.gather(*tasks, desc="Processing messages"),  # type: ignore
        )
    else:
        responses = await asyncio.gather(*tasks)

    return responses


def parse_anthropic_completion(response: Message | None) -> str | None:
    if response is None:
        return None
    try:
        first_content = response.content[0]
        if first_content.type == "text":
            return first_content.text
        else:
            return None
    except (AttributeError, IndexError):
        return None
