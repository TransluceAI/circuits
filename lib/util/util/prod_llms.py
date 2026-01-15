"""
At some point we'll want to do a refactor to support different types of provider/key swapping
due to different scenarios. However, this'll probably be a breaking change, which is why I'm
not doing it now.

- mengk
"""

import asyncio
import traceback
from typing import Any, Callable, Protocol, TypedDict

from env_util import ENV
from util import anthropic, openai, together
from util.llm_cache import LLMCache
from util.types import ChatMessage


class AsyncCompletionGetter(Protocol):
    async def __call__(
        self,
        client: Any,
        messages_list: list[list[ChatMessage]],
        model_name: str,
        *,
        max_new_tokens: int,
        temperature: float,
        logprobs: bool,
        top_logprobs: int | None,
        max_concurrency: int | None,
        timeout: float,
        use_tqdm: bool,
        semaphore: asyncio.Semaphore | None,
    ) -> list[Any]: ...


class ProviderConfig(TypedDict):
    keys: list[str]
    current_key_index: int
    async_client: openai.AsyncOpenAI | anthropic.AsyncAnthropic
    async_completion_getter: AsyncCompletionGetter
    completion_parser: Callable[[Any], str | None]
    models: dict[str, str]


class LLMManager:
    def __init__(
        self,
        default_provider: str = "anthropic",
        max_concurrency: int = 100,
        rotate_providers: bool = True,
        timeout: float = 5.0,
        use_cache: bool = False,
    ):
        if ENV.ANTHROPIC_API_KEY is None:
            raise Exception("Missing Anthropic API key; check your .env")
        if ENV.OPENAI_API_KEY is None:
            raise Exception("Missing OpenAI API key; check your .env")

        self.cache = LLMCache() if use_cache else None

        self.providers: dict[str, ProviderConfig] = {
            "anthropic": {
                "keys": [ENV.ANTHROPIC_API_KEY],
                "current_key_index": 0,
                "async_client": anthropic.get_anthropic_client_async(),
                "async_completion_getter": anthropic.get_anthropic_chat_completions_async,
                "completion_parser": anthropic.parse_anthropic_completion,
                "models": {
                    "smart": "claude-3-5-sonnet-20241022",
                    "fast": "claude-3-haiku-20240307",
                },
            },
            "openai": {
                "keys": [ENV.OPENAI_API_KEY],
                "current_key_index": 0,
                "async_client": openai.get_openai_client_async(),
                "async_completion_getter": openai.get_openai_chat_completions_async,
                "completion_parser": openai.parse_openai_completion,
                "models": {
                    "smart": "gpt-4o-2024-08-06",
                    "fast": "gpt-4o-mini-2024-07-18",
                    "reasoning_fast": "o1-mini",
                    "reasoning_smart_preview": "o1-preview",
                    "reasoning_smart": "o1",
                },
            },
            "together": {
                "keys": [ENV.TOGETHER_API_KEY],
                "current_key_index": 0,
                "async_client": together.get_together_client_async(),
                "async_completion_getter": together.get_together_chat_completions_async,
                "completion_parser": together.parse_together_completion,
                "models": {
                    "smart": "deepseek-ai/DeepSeek-V3t",
                    "fast": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                },
            },
            # Add more providers here as needed
        }

        if default_provider not in self.providers:
            raise ValueError(f"Invalid default provider: {default_provider}")

        # Reorder provider_order to put default_provider first
        if rotate_providers:
            self.provider_order = [default_provider] + [
                p for p in self.providers.keys() if p != default_provider
            ]
        else:
            self.provider_order = [default_provider]
        self.current_provider_index = 0
        self.provider = self.providers[self.provider_order[self.current_provider_index]]

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = timeout

    async def get_completions(
        self,
        messages_list: list[list[ChatMessage]],
        model_category: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        max_concurrency: int | None = None,
        timeout: float | None = None,
        model_id: str | None = None,
    ) -> list[str | None]:
        """
        If max_concurrency is None, use LLManager.semaphore to manage concurrency
        """
        if timeout is None:
            timeout = self.timeout

        results: list[str | None] = [None] * len(messages_list)

        while True:
            try:
                client = self.provider["async_client"]
                client.api_key = self.provider["keys"][self.provider["current_key_index"]]
                model_name = self.provider["models"].get(model_category)

                # This should trigger if reasoning is used for Anthropic
                if model_name is None:
                    raise Exception(
                        f"Model category '{model_category}' not found in provider '{self.provider_order[self.current_provider_index]}'"
                    )

                # Track uncached messages
                uncached_indices: list[int] = list(range(len(messages_list)))
                uncached_messages: list[list[ChatMessage]] = messages_list.copy()

                # Check cache if available
                if self.cache is not None:
                    uncached_indices = []
                    uncached_messages = []
                    hits = 0

                    for i, messages in enumerate(messages_list):
                        cached_result = self.cache.get(messages, model_name)
                        if cached_result is not None:
                            results[i] = cached_result
                            hits += 1
                        else:
                            uncached_indices.append(i)
                            uncached_messages.append(messages)

                    misses = len(messages_list) - hits
                    print(f"[LLM cache] {model_name}: {hits} hits, {misses} misses")

                # Get completions for uncached messages
                if uncached_messages:
                    completions = await self.provider["async_completion_getter"](
                        client,
                        uncached_messages,
                        model_name,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        logprobs=False,
                        top_logprobs=None,
                        max_concurrency=max_concurrency,
                        timeout=timeout,
                        use_tqdm=len(uncached_messages) > 20,
                        semaphore=self.semaphore,
                    )

                    # If all completions are None, rotate keys and swap provider
                    # TODO(kevin): this is not a great key swapping condition; we should refactor at some point.
                    if all(completion is None for completion in completions):
                        raise Exception("All completions are None")

                    # Parse completions and update cache
                    parser = self.provider["completion_parser"]
                    for i, (messages, completion) in enumerate(zip(uncached_messages, completions)):
                        parsed = parser(completion)
                        if parsed is not None and self.cache is not None:
                            self.cache.set(messages, model_name, parsed)
                        results[uncached_indices[i]] = parsed

                break  # Break to return the results
            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())
                if not self._rotate_keys_and_swap_provider():
                    print("All keys and providers exhausted.")
                    break  # Break to return the results as is

        return results

    def _rotate_keys_and_swap_provider(self) -> bool:
        """
        Rotate to the next API key for the current provider.
        If all keys for the current provider are exhausted, move to the next provider.
        Return True if there is a new key/provider to try, False if all are exhausted.
        """

        # Move to the next key index
        self.provider["current_key_index"] += 1

        if self.provider["current_key_index"] < len(self.provider["keys"]):
            # There are more keys in this provider
            provider_name = self.provider_order[self.current_provider_index]
            print(f"Switched to next key for provider '{provider_name}'.")
            return True
        else:
            # No more keys in the current provider, reset and move to next provider
            self.provider["current_key_index"] = 0  # Reset key index for this provider
            self.current_provider_index += 1
            if self.current_provider_index < len(self.provider_order):
                # Move to next provider
                provider_name = self.provider_order[self.current_provider_index]
                self.provider = self.providers[provider_name]
                print(f"Switched to next provider '{provider_name}'.")
                return True
            else:
                # All providers and their keys have been exhausted
                print("All providers and their keys have been exhausted.")
                return False


async def get_llm_completions_async(
    messages_list: list[list[ChatMessage]],
    model_category: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
    timeout: float = 5.0,
    default_provider: str = "anthropic",
    use_cache: bool = False,
):
    print("WARN: util/prod_llms.py is deprecated; migrate to util/llm/prod_llms.py instead.")
    llm_manager = LLMManager(
        default_provider=default_provider,
        max_concurrency=max_concurrency,
        use_cache=use_cache,
    )
    return await llm_manager.get_completions(
        messages_list, model_category, max_new_tokens, temperature, max_concurrency, timeout
    )
