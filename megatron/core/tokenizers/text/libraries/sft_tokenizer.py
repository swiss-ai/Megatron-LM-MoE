# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False


# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
# fmt: on


IGNORE_INDEX = -100


def to_apertus_messages(conversation: List[Dict]) -> Tuple[List[Dict], Optional[list], bool]:
    """Turn a conversation into Apertus chat-template input: (messages, tools, enable_thinking).

    Accepts plain {"role", "content": str} messages ("from"/"value" too) and the Apertus SFT-mix
    schema. The mix keeps tools and the deliberation flag in a leading "developer" message,
    which the template rejects: it builds the developer block itself from `tools` and
    `enable_thinking`. Nested content (user `parts`, assistant `blocks` with thoughts,
    tool_calls, tool_outputs) is passed through, minus nulls and the empty {"name": ""}
    placeholder calls/outputs. A message's "train" flag is kept.
    """
    messages, tools, enable_thinking = [], None, False
    for message in conversation:
        role = message.get("role", message.get("from", "")).lower()
        content = message.get("content", message.get("value", ""))
        is_mapping = isinstance(content, dict)
        if is_mapping and content.get("has_thinking"):
            enable_thinking = True
        if role == "developer":
            raw_tools = content.get("tools") if is_mapping else None
            if raw_tools:
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            continue
        if not is_mapping:
            content = content or ""
        elif role == "user":
            parts = [
                {"type": p["type"], "text": p.get("text") or ""} for p in content.get("parts") or []
            ]
            if not parts and content.get("text"):
                parts = [{"type": "text", "text": content["text"]}]
            content = {"parts": parts}
        elif role == "assistant":
            blocks = []
            for b in content.get("blocks") or []:
                block = {"type": b["type"], "text": b.get("text") or ""}
                if b["type"] == "thoughts" and block["text"]:
                    enable_thinking = True
                elif b["type"] == "tool_calls":
                    block["calls"] = [
                        {"name": c["name"], "arguments": c.get("arguments") or "{}"}
                        for c in b.get("calls") or []
                        if c.get("name")
                    ]
                elif b["type"] == "tool_outputs":
                    block["outputs"] = [
                        {"name": o.get("name") or "", "output": o.get("output") or ""}
                        for o in b.get("outputs") or []
                        if o.get("name") or o.get("output")
                    ]
                blocks.append(block)
            if not blocks and content.get("text"):
                blocks = [{"type": "response", "text": content["text"]}]
            content = {"blocks": blocks}
        else:  # system, tool
            content = content.get("text") or ""
        converted = {"role": role, "content": content}
        if "train" in message:
            converted["train"] = message["train"]
        messages.append(converted)
    return messages, tools, enable_thinking


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None


class SFTTokenizer:
    """SFT Tokenizer."""

    def __init__(self, tokenizer_path: str, prompt_format: str):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        if HAVE_TRANSFORMERS:
            # Currently, only HuggingFace tokenizers are supported.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "identity":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=identity_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "apertus":
            # Template: <s><|system_start|>...<|system_end|><|developer_start|>...<|developer_end|>
            #   <|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>
            apertus_template = transformers.AutoTokenizer.from_pretrained(
                "swiss-ai/Apertus-8B-Instruct-2509"
            ).chat_template
            self._prompt_config = PromptConfig(
                # <|assistant_start|> is rendered by the assistant turn; don't train on it.
                assistant_prefix_len=1,
                # <pad> (3); must differ from eos <|assistant_end|> (68), or SFTDataset masks EOS.
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=apertus_template,
                # The template emits {{ bos_token }} itself.
                has_bos=True,
                has_system_role=True,
            )
            # Mask boundaries; the tokenizer's sft_* fields (create_instruct.py) take precedence.
            begin = tokenizer.init_kwargs.get("sft_assistant_begin_sequence") or [
                tokenizer.convert_tokens_to_ids("<|assistant_start|>")
            ]
            eot = tokenizer.init_kwargs.get("sft_eot_token") or [
                tokenizer.convert_tokens_to_ids("<|assistant_end|>")
            ]
            self._apertus_assistant_start, self._apertus_assistant_end = begin[0], eot[0]
            assert tokenizer.unk_token_id not in (begin[0], eot[0]), (
                f"{tokenizer_path} has no <|assistant_start|>/<|assistant_end|> tokens"
            )
        elif prompt_format == "default":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        if self._prompt_format == "apertus":
            return self._tokenize_conversation_apertus(
                conversation, return_target, add_generation_prompt
            )

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=self._prompt_config.custom_chat_template,
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # When using the default prompt format, we do not replace any tokens with IGNORE_INDEX.
        # Instead, all token losses will be used for simplicity.
        if self._prompt_format == "default":
            return tokens, target

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):

            if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if turn["role"].lower() == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user", "tool"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            else:
                raise ValueError("Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def _tokenize_conversation_apertus(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Apertus tokenization, ported from swiss-ai/multimodal-data chat_preprocess.py
        (style="apertus", add_bos=False, add_eos=True as in its SFTChatDataset).

        The Apertus template emits <s> + system + developer blocks on every render, so the
        per-turn re-render used for the other formats cannot align turns. Instead the whole
        conversation is rendered once and the mask is built by scanning for
        <|assistant_start|> ... <|assistant_end|>: tokens after the start token up to and
        including the end token are trained, unless that assistant turn has "train": False.
        Rows in the Apertus SFT-mix schema are converted first (see to_apertus_messages).
        """
        chat, tools, enable_thinking = to_apertus_messages(conversation)
        # Render to text, then encode: the template emits {{ bos_token }} itself, and this
        # returns plain ids across transformers versions (v5 returns a BatchEncoding).
        text = self._tokenizer.apply_chat_template(
            chat,
            tools=tools,
            enable_thinking=enable_thinking,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self._prompt_config.custom_chat_template,
        )
        tokens = np.asarray(
            self._tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64
        )
        if not return_target:
            return tokens

        train_flags = [m.get("train", True) for m in chat if m["role"] == "assistant"]
        mask = np.zeros(len(tokens), dtype=bool)
        turn_idx, in_assistant, should_train = 0, False, True
        for i, tok in enumerate(tokens.tolist()):
            # Mask first (before state changes): the start token itself is never trained.
            if in_assistant and should_train:
                mask[i] = True
            if tok == self._apertus_assistant_start:
                in_assistant = True
                should_train = train_flags[turn_idx] if turn_idx < len(train_flags) else True
            if tok == self._apertus_assistant_end:
                if in_assistant:
                    turn_idx += 1
                in_assistant = False

        # BOS is never trained; EOS is appended (and trained) if the render doesn't end with it.
        if len(tokens) and tokens[0] == self._tokenizer.bos_token_id:
            mask[0] = False
        eos_id = self._tokenizer.eos_token_id
        ends_with_eos = len(tokens) > 0 and tokens[-1] == eos_id
        if not add_generation_prompt and eos_id is not None and not ends_with_eos:
            tokens = np.append(tokens, eos_id)
            mask = np.append(mask, True)

        target = np.where(mask, tokens, IGNORE_INDEX)

        return tokens, target

    def text_to_ids(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(
                text, return_target=False, add_generation_prompt=True
            ).tolist()

        return self._tokenizer.encode(text)

    def tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_text(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def ids_to_tokens(self):
        """Converts ids to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def text_to_tokens(self):
        """Converts text to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def tokens_to_text(self):
        """Converts tokens to text."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    def add_special_tokens(self):
        """Add special tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    @property
    def pad_id(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def bos_id(self):
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
