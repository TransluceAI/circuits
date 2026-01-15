from __future__ import annotations

from abc import abstractmethod
from html import escape
from typing import TYPE_CHECKING, Any, Generator, cast

from pydantic import BaseModel, field_validator
from util.types import ChatMessage, GenerateOutput

# Prevents circular import (which I think is unavoidable / annoying to resolve)
if TYPE_CHECKING:
    from util.subject import Subject

STRIPPED_LLAMA_CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'


def strip_array_in_place(arr: list[Any], value: Any) -> list[Any]:
    """
    Strips `value` from beginning and end of `arr`.
    Performs operation in place.
    """

    while arr and arr[0] == value:
        arr.pop(0)
    while arr and arr[-1] == value:
        arr.pop()

    return arr


def strip_starting_at_rindex_in_place(arr: list[Any], value: Any) -> list[Any]:
    """
    Strips everything including and after the final occurrence of `value` within `arr`.
    """

    try:
        rindex = arr[::-1].index(value)
        index = len(arr) - 1 - rindex
        del arr[index:]
    except ValueError:
        pass

    return arr


class ModelInput(BaseModel):
    use_chat_format: bool

    @abstractmethod
    def tokenize(self, subject: "Subject", *_: Any, **__: Any) -> list[int]:
        pass

    def token_strs(self, subject: "Subject") -> list[str]:
        return [subject.decode(x) for x in self.tokenize(subject)]

    def is_empty(self, subject: "Subject") -> bool:
        return len(self.tokenize(subject)) == 0

    ###########
    # Display #
    ###########

    def to_str(self, subject: "Subject") -> str:
        return subject.decode(self.tokenize(subject))  # type: ignore

    def pretty_print_tokens(
        self, subject: "Subject", token_highlights: list[tuple[int, float, str]] | None = None
    ):
        toks = self.tokenize(subject)

        # Check if we're in a Jupyter environment
        try:
            from IPython.display import HTML, display  # type: ignore

            in_jupyter = True
        except ImportError:
            in_jupyter = False

        # If so, pretty print tokens in Jupyter
        if in_jupyter:
            # If token_highlights provided, prepare highlights and metadata
            token_sums = None
            token_metadata = None
            if token_highlights is not None:
                token_sums = [0.0] * len(toks)
                token_metadata = [[] for _ in range(len(toks))]
                for token_idx, value, metadata in token_highlights:
                    token_sums[token_idx] += value
                    token_metadata[token_idx].append(metadata)

                # Normalize values to [0, 1] range
                max_sum = max(token_sums)
                if max_sum > 0:
                    token_sums = [v / max_sum for v in token_sums]
                else:
                    token_sums = [0.0 for _ in token_sums]

            html_tokens: list[str] = []
            token_info_div = (
                '<div id="token-info" style="display: block; width: 100%; '
                "margin: 1px 0; padding: 10px; overflow-y: auto; max-height: 300px; "
                "border: 1px solid #ccc; border-radius: 2px; background: #fff; "
                'font-family: monospace; color: #333; margin-top: 10px;">'
                "Hover over tokens to see neuron information"
                "</div>"
            )

            for i, tok in enumerate(toks):
                decoded = repr(subject.decode(tok)).strip("'")
                # Add background color if we have highlights
                if token_sums is not None:
                    opacity = token_sums[i]
                    bg_style = f"background: rgb(255, {255 - 55*opacity}, {255 - 255*opacity});"
                else:
                    bg_style = "background: #fff;"

                # Create mouseover events for neuron information
                if token_metadata and token_metadata[i]:
                    metadata_str = "<br/><br/>".join(
                        "&nbsp;-&nbsp;" + escape(m) for m in token_metadata[i]
                    )
                    mouseover = f"""onmouseover="document.getElementById('token-info').innerHTML = '<b>Token {i}:</b><br/><br/>{metadata_str}'" """
                    mouseout = """onmouseout="document.getElementById('token-info').innerHTML = 'Hover over tokens to see neuron information'" """
                else:
                    mouseover = ""
                    mouseout = ""

                html_tokens.append(
                    f'<div style="display: inline-block; margin: 1px; padding: 1px 2px; '
                    f'border: 1px solid #ccc; border-radius: 2px; {bg_style}" {mouseover} {mouseout}>'
                    f'<small style="color: #888; display: block">{i}</small>'
                    f"{decoded}</div>"
                )
            html = (
                '<div style="font-family: monospace; line-height: 1; color: #333">'
                + "".join(html_tokens)
                + "</div>"
                + token_info_div
            )

            display(HTML(html))  # type: ignore
            return None

        # Otherwise, return original plain text formatting
        else:
            return "\n".join(
                [f"{i:>3}: {repr(subject.decode(x)).strip('\'')}" for i, x in enumerate(toks)]
            )


class IdsInput(ModelInput):
    input_ids: list[int] | None = None
    use_chat_format: bool = (
        False  # Input IDs are interpreted literally, so use_chat_format is set to false
    )

    def tokenize(self, *_: Any, **__: Any) -> list[int]:
        if self.input_ids is None:
            raise ValueError("token_ids must be set")
        return self.input_ids


class ChatInput(ModelInput):
    system_prompt: list[ChatMessage] | None
    conversation: list[ChatMessage]
    seed_response: str | None = None  # Allows you to "put words in the model's mouth"
    use_chat_format: bool

    @classmethod
    def from_dicts(cls, dicts: list[ChatMessage]):
        system_prompt: list[ChatMessage] = []
        conversation: list[ChatMessage] = []

        for d in dicts:
            if d["role"] == "system":
                system_prompt.append(d)
            else:
                conversation.append(d)

        return cls(system_prompt=system_prompt, conversation=conversation, use_chat_format=True)

    def to_dicts(self) -> list[ChatMessage]:
        return [*(self.system_prompt or []), *self.conversation]

    ################
    # Tokenization #
    ################

    def is_empty(self, subject: Subject) -> bool:
        return (
            len(self.conversation) == 0
            and self.system_prompt is None
            and self.seed_response is None
        )

    def tokenize(self, subject: Subject, trim_bos: bool = False) -> list[int]:
        if self.is_empty(subject):
            return []

        if self.use_chat_format:
            assert (
                subject.is_chat_model
            ), "Cannot tokenize for a chat model if subject is not a chat model"

            if len(self.conversation) > 0:
                last_role = self.conversation[-1]["role"]
            elif self.system_prompt is not None:
                last_role = "system"
            else:
                last_role = None

            # Append messages based on provided inputs
            messages: list[ChatMessage] = []
            if self.system_prompt is not None:
                messages.extend(self.system_prompt)
            messages.extend(self.conversation)
            if self.seed_response is not None:
                messages.append({"role": "assistant", "content": self.seed_response})

            # Tokenize messages
            toks: list[int] = subject.tokenizer.apply_chat_template(  # type: ignore
                cast(list[dict[str, str]], messages),
                add_generation_prompt=last_role == "user" and not self.seed_response,  # type: ignore
                chat_template=STRIPPED_LLAMA_CHAT_TEMPLATE,  # type: ignore
            )
            if self.seed_response:
                # Strip everything starting at last EOS token (sometimes there are extraneous tokens)
                strip_starting_at_rindex_in_place(toks, subject.tokenizer.eos_token_id)
        else:
            # Simply use system prompt
            assert (
                len(self.conversation) == 1
            ), f"If `subject.is_chat_model` is False, we tokenize only the first conversation message. Got too many conversation messages: {self.conversation}"
            assert (
                self.system_prompt is None and self.seed_response is None
            ), "If `subject.is_chat_model` is False, system_prompt and seed_response must be None."

            toks: list[int] = subject.tokenizer(self.conversation[0]["content"])["input_ids"]  # type: ignore

        if trim_bos:
            strip_array_in_place(toks, subject.tokenizer.bos_token_id)

        return toks

    ##############
    # Validation #
    ##############

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: list[ChatMessage] | None):
        if v:
            cls._validate_chat(v, ends_with_roles=["system"])
        return v

    @field_validator("conversation")
    @classmethod
    def validate_conversation(cls, v: list[ChatMessage]):
        cls._validate_chat(v, ends_with_roles=["user"])
        return v

    @staticmethod
    def _validate_chat(chat: list[ChatMessage], ends_with_roles: list[str]):
        """
        Checks that:
        - Role/content formatting is correct
        - Chat starts with system or user
        - Chat alternates between system/user and assistant
        - Chat ends with one of the roles in ends_with_roles
        """

        def _validate_prompt_block(block: ChatMessage):
            assert "role" in block, "role must exist in each block"
            assert "content" in block, "content must exist in each block"
            assert block["role"] in [
                "system",
                "user",
                "assistant",
            ], "role must be either 'user' or 'assistant'"

        role = None
        for block in chat:
            _validate_prompt_block(block)
            if role is None:
                assert block["role"] in [
                    "system",
                    "user",
                ], "Conversation must start with system or user"
                role = block["role"]
            else:
                assert (
                    block["role"] != role
                ), "Conversation must alternate roles between system/user and assistant"
                role = block["role"]

        if len(chat) > 0:
            assert (
                chat[-1]["role"] in ends_with_roles
            ), f"Conversation must end with any of {ends_with_roles}"

    ########################
    # Equality and Hashing #
    ########################

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ChatInput):
            return (
                self.system_prompt == other.system_prompt
                and self.conversation == other.conversation
                and self.seed_response == other.seed_response
                and self.use_chat_format == other.use_chat_format
            )
        return False

    def __hash__(self):
        return hash(
            (
                (
                    tuple(tuple(sorted(d.items())) for d in self.system_prompt)
                    if self.system_prompt
                    else None
                ),
                tuple(tuple(sorted(d.items())) for d in self.conversation),
                self.seed_response,
            )
        )

    ############
    # Indexing #
    ############

    def slice(
        self, system_prompt_slice: slice | None = None, conversation_slice: slice | None = None
    ) -> "ChatInput":
        """
        Returns a new ChatInput instance with system_prompt and conversation sliced according to the given slices.

        Args:
            system_prompt_slice: A slice object to apply to the system_prompt list.
            conversation_slice: A slice object to apply to the conversation list.

        Returns:
            A new ChatInput instance with sliced system_prompt and conversation.
        """

        if system_prompt_slice is None:
            system_prompt_slice = slice(None)
        if conversation_slice is None:
            conversation_slice = slice(None)

        return ChatInput(
            system_prompt=self.system_prompt[system_prompt_slice] if self.system_prompt else None,
            conversation=self.conversation[conversation_slice],
            seed_response=self.seed_response,
            use_chat_format=self.use_chat_format,
        )


class ChatConversation(ChatInput):
    @field_validator("conversation")
    @classmethod
    def validate_conversation(cls, v: list[ChatMessage]):
        """
        Conversations don't have to end with anything in particular.
        Overrides the parent class validator, where conversations must end with "user".
        """

        cls._validate_chat(v, ends_with_roles=["user", "assistant"])
        return v

    def add_messages(self, messages: list[ChatMessage]):
        self.conversation.extend(messages)
        self._validate_chat(self.conversation, ends_with_roles=["user", "assistant"])

    def send_message(
        self,
        subject: "Subject",
        message: str | None,
        seed_response: str | None = None,
        neuron_interventions: dict[tuple[int, int, int], float] | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        stream: bool = False,
        verbose: bool = False,
    ) -> GenerateOutput | Generator[int | GenerateOutput, None, None]:
        if subject.is_chat_model:
            # If message is None, we'll be regenerating based on the last user message
            if message is None:
                assert seed_response is None, f"Cannot provide seed_response when message is None"

                # Remove messages until the last one is a user message
                while len(self.conversation) > 0 and self.conversation[-1]["role"] != "user":
                    self.conversation.pop()
            else:
                # Add user message to conversation
                self.add_messages([{"role": "user", "content": message}])
        else:
            assert (
                seed_response is None
            ), f"Cannot provide seed_response when subject is not a chat model"
            assert message is not None, f"Must provide message when subject is not a chat model"

            # Add a message if the conversation is empty
            if len(self.conversation) == 0:
                self.add_messages([{"role": "user", "content": message}])
            # Otherwise, add user message to existing message
            else:
                cur_message = self.conversation[0]["content"]
                self.conversation = [{"role": "user", "content": cur_message + message}]

        # Reload CC into CI to validate that it's formatted correctly as an input
        # Because CI requires that the last message is from the user
        # Also add seed_response if it exists
        ci = ChatInput(**(self.model_dump() | {"seed_response": seed_response}))

        # Generate response and update chat input
        response = subject.generate(
            ci,
            neuron_interventions=neuron_interventions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
            verbose=verbose,
        )

        def _postprocess(final_response: GenerateOutput):
            if subject.is_chat_model:
                self.add_messages(
                    [
                        {
                            "role": "assistant",
                            "content": (seed_response or "") + final_response.continuations[0],
                        }
                    ]
                )
            else:
                self.conversation[0]["content"] += final_response.continuations[0]

        if stream:
            assert isinstance(response, Generator), "Stream must return a generator"

            def _generator():
                nonlocal response

                for update in response:
                    if isinstance(update, int):
                        yield update
                    else:
                        # Text is done streaming; record the final output
                        response = update

                if isinstance(response, GenerateOutput):
                    _postprocess(response)
                    yield response

            return _generator()
        else:
            assert isinstance(
                response, GenerateOutput
            ), "Non-streamed generation must return a GenerateOutput"

            _postprocess(response)
            return response


def make_chat_conversation(system_prompt: str | None = None):
    return ChatConversation(
        system_prompt=(
            [{"role": "system", "content": system_prompt}] if system_prompt is not None else None
        ),
        conversation=[],
        use_chat_format=True,
    )


def make_chat_input(
    system_prompt: str | None,
    message: str,
    seed_response: str | None = None,
    use_chat_format: bool = True,
) -> ChatInput:
    # Parse system prompt
    system_prompt_list: list[ChatMessage] | None = None
    if system_prompt:
        if use_chat_format:
            system_prompt_list = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]
        else:  # Simulate a system prompt by acking a command
            system_prompt_list = [
                {
                    "role": "user",
                    "content": f'{system_prompt}\n\nReply with only "OK" to acknowledge.',
                },
                {
                    "role": "assistant",
                    "content": "OK",
                },
            ]

    conversation: list[ChatMessage] = [{"role": "user", "content": message}]

    return ChatInput(
        system_prompt=system_prompt_list,
        conversation=conversation,
        seed_response=seed_response,
        use_chat_format=use_chat_format,
    )
