import asyncio
import json
import traceback
from contextlib import nullcontext
from functools import partial
from typing import cast
from uuid import uuid4

import backoff
import tiktoken
from backoff.types import Details
from env_util import ENV
from openai import AsyncOpenAI, BadRequestError, OpenAI
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tqdm.asyncio import tqdm_asyncio
from util.types import ChatMessage

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
MAX_EMBEDDING_TOKENS = 8000


def _print_backoff_message(e: Details):
    print(
        f"OpenAI backing off for {e['wait']:.2f}s due to {e['exception'].__class__.__name__}"  # type: ignore
    )


class CompletionTooLongException(Exception):
    pass


@backoff.on_exception(
    backoff.expo,
    exception=(Exception,),
    giveup=lambda e: isinstance(e, BadRequestError),  # Don't retry on BadRequestError
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def get_openai_chat_completion_streaming_async(
    client: AsyncOpenAI,
    messages: list[ChatMessage],
    model_name: str,
    temperature: float = 1.0,
    timeout: float = 30.0,
):
    """
    Stream a chat completion from OpenAI's API.

    Args:
        client (AsyncOpenAI): The async OpenAI client to use for the request
        messages (list[ChatMessage]): The conversation messages to send to the API
        model_name (str): Name of the OpenAI model to use (e.g. "gpt-4")
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        timeout (float, optional): Timeout in seconds. Defaults to 30.0.

    Returns:
        AsyncGenerator[str, None]: An async generator that yields chunks of the response text as they arrive

    Raises:
        TimeoutError: If the request exceeds the timeout duration
        Exception: If the API request fails after max retries
    """
    async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():  # type: ignore
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[cast(ChatCompletionMessageParam, message) for message in messages],
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


@backoff.on_exception(
    backoff.expo,
    exception=(Exception,),
    giveup=lambda e: isinstance(e, BadRequestError),  # Don't retry on BadRequestError
    max_tries=3,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_openai_chat_completion_async(
    client: AsyncOpenAI,
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
            messages=[cast(ChatCompletionMessageParam, message) for message in messages],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        # If the completion is empty and was truncated (likely due to too much reasoning), raise an exception
        if output.choices[0].finish_reason == "length" and (
            (parsed := parse_openai_completion(output)) is None or len(parsed) == 0
        ):
            raise CompletionTooLongException(
                "Completion empty due to truncation. Consider increasing max_new_tokens."
            )

        return output


def get_openai_client_sync() -> OpenAI:
    if ENV.OPENAI_API_KEY is None:
        raise Exception("Missing OpenAI API key; check your .env")
    return OpenAI(api_key=ENV.OPENAI_API_KEY)


def get_openai_client_async() -> AsyncOpenAI:
    if ENV.OPENAI_API_KEY is None:
        raise Exception("Missing OpenAI API key; check your .env")
    return AsyncOpenAI(api_key=ENV.OPENAI_API_KEY)


async def get_openai_chat_completions_async(
    client: AsyncOpenAI,
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
    print("WARN: util/openai.py is deprecated; migrate to util/llm/prod_llms.py instead.")

    base_func = partial(
        _get_openai_chat_completion_async,
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
            list[ChatCompletion | None],
            await tqdm_asyncio.gather(*tasks, desc="Processing messages"),  # type: ignore
        )
    else:
        responses = await asyncio.gather(*tasks)

    return responses


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=10,
    factor=2.0,
    on_backoff=_print_backoff_message,
)
async def _get_openai_embeddings_async_one_batch(
    client: AsyncOpenAI, texts_batch: list[str], model_name: str, dimensions: int | None
):
    response = await client.embeddings.create(
        model=model_name,
        input=texts_batch,
        dimensions=dimensions if dimensions is not None else NOT_GIVEN,
    )
    return [data.embedding for data in response.data]


async def get_openai_embeddings_async(
    client: AsyncOpenAI,
    texts: list[str],
    model_name: str = "text-embedding-3-large",
    dimensions: int | None = 1536,
    max_concurrency: int = 100,
) -> list[list[float] | None]:
    """
    Asynchronously get embeddings for a list of texts using OpenAI's embedding model.
    This function uses tiktoken for tokenization, truncates at 8000 tokens, and prints a warning if truncation occurs.
    Concurrency is limited using a semaphore.
    """

    if model_name != "text-embedding-3-large":
        assert dimensions == None, f"{model_name} does not have a variable dimension size"

    # Tokenize and truncate texts
    tokenizer = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    truncated_texts: list[str] = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_EMBEDDING_TOKENS:
            print(
                f"Warning: Text at index {i} has been truncated from {len(tokens)} to {MAX_EMBEDDING_TOKENS} tokens."
            )
            tokens = tokens[:MAX_EMBEDDING_TOKENS]
        truncated_texts.append(tokenizer.decode(tokens))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(texts_batch: list[str]):
        async with semaphore:
            try:
                return await _get_openai_embeddings_async_one_batch(
                    client, texts_batch, model_name, dimensions
                )
            except Exception as e:
                print(f"Error in fetch_embeddings: {e}. Returning None.")
                return [None] * len(texts_batch)

    # Create batches of 1000 texts (OpenAI's current limit per request)
    batches = [truncated_texts[i : i + 1000] for i in range(0, len(truncated_texts), 1000)]

    # Run tasks concurrently
    tasks = [limited_task(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten the results
    embeddings = [embedding for batch_result in results for embedding in batch_result]

    return embeddings


def get_openai_embeddings_sync(
    client: OpenAI,
    texts: list[str],
    model_name: str = "text-embedding-3-large",
    dimensions: int | None = 1536,
) -> list[list[float] | None]:
    """
    Synchronously get embeddings for a list of texts using OpenAI's embedding model.
    This function uses tiktoken for tokenization and truncates at 8000 tokens.
    """
    # Tokenize and truncate texts
    tokenizer = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    truncated_texts: list[str] = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_EMBEDDING_TOKENS:
            print(
                f"Warning: Text at index {i} has been truncated from {len(tokens)} to {MAX_EMBEDDING_TOKENS} tokens."
            )
            tokens = tokens[:MAX_EMBEDDING_TOKENS]
        truncated_texts.append(tokenizer.decode(tokens))

    # Process in batches of 1000
    embeddings: list[list[float] | None] = []
    for i in range(0, len(truncated_texts), 1000):
        batch = truncated_texts[i : i + 1000]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=dimensions if dimensions is not None else NOT_GIVEN,
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error in get_openai_embeddings_sync: {e}")
            embeddings.extend([None] * len(batch))

    return embeddings


def parse_openai_completion(response: ChatCompletion | None) -> str | None:
    if response is None:
        return None
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError):
        return None


##############
# Finetuning #
##############


def _prepare_jsonl_file(dataset: list[tuple[str, str]], prefix: str) -> tuple[str, str]:
    """Helper function to prepare and save dataset to JSONL file"""
    data = [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        }
        for prompt, completion in dataset
    ]

    fname = f"_tmp_{prefix}_{str(uuid4())}_gitignore.jsonl"
    with open(fname, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    return fname


def fine_tune_model_openai(
    client: OpenAI,
    train_dataset: list[tuple[str, str]],
    test_dataset: list[tuple[str, str]] | None = None,
    model_name: str = "gpt-4o-mini-2024-07-18",
):
    # Prepare files
    train_fname = _prepare_jsonl_file(train_dataset, "training_data")
    train_file = client.files.create(file=open(train_fname, "rb"), purpose="fine-tune")

    # Handle test dataset if provided
    test_file = None
    if test_dataset:
        test_fname = _prepare_jsonl_file(test_dataset, "test_data")
        test_file = client.files.create(file=open(test_fname, "rb"), purpose="fine-tune")

    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=test_file.id if test_file else None,
        model=model_name,
    )

    return job


def estimate_openai_prompt_tokens(messages: list[ChatMessage]) -> int:
    """
    Approximately-correct implementation adapted from this documentation:
    https://platform.openai.com/docs/guides/chat/introduction
    """

    encoding = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <|im_start|>{role/name}\n{content}<|im_end|>\n
        num_tokens += len(encoding.encode(message["content"], allowed_special="all"))

    num_tokens += 2  # every reply is primed with <|im_start|>assistant

    return num_tokens


def num_openai_tokens(sequence: str) -> int:
    encoding = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    return len(encoding.encode(sequence, allowed_special="all"))
