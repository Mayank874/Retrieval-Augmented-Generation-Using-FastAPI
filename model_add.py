from haystack.nodes import PromptModelInvocationLayer
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: Union[str, os.PathLike],
                 max_length: Optional[int] = 128,
                 max_context: Optional[int] = 2048,
                 n_parts: Optional[int] = -1,
                 seed: Optional[int] = 1337,
                 f16_kv: Optional[bool] = True,
                 logits_all: Optional[bool] = False,
                 vocab_only: Optional[bool] = False,
                 use_mmap: Optional[bool] = True,
                 use_mlock: Optional[bool] = False,
                 embedding: Optional[bool] = False,
                 n_threads: Optional[int] = None,
                 n_batch: Optional[int] = 512,
                 last_n_tokens_size: Optional[int] = 64,
                 lora_base: Optional[str] = None,
                 lora_path: Optional[str] = None,
                 verbose: Optional[bool] = True,
                 **kwargs):
        if not model_name_or_path:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path
        self.max_context = max_context
        self.max_length = max_length
        self.n_parts = n_parts
        self.seed = seed
        self.f16_kv = f16_kv
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.embedding = embedding
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.lora_base = lora_base
        self.lora_path = lora_path
        self.verbose = verbose

        self.model = Llama(model_path=model_name_or_path,
                           n_ctx=max_context,
                           n_parts=n_parts,
                           seed=seed,
                           f16_kv=f16_kv,
                           logits_all=logits_all,
                           vocab_only=vocab_only,
                           use_mmap=use_mmap,
                           use_mlock=use_mlock,
                           embedding=embedding,
                           n_threads=n_threads,
                           n_batch=n_batch,
                           last_n_tokens_size=last_n_tokens_size,
                           lora_base=lora_base,
                           lora_path=lora_path,
                           verbose=verbose)

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure prompt length fits within max token limit."""
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be of type str but got {type(prompt)}")

        context_length = self.model.n_ctx()
        tokenized_prompt = self.model.tokenize(bytes(prompt, 'utf-8'))
        if len(tokenized_prompt) + self.max_length > context_length:
            logger.warning(
                "Prompt truncated to fit within max token limit.")
            return bytes.decode(self.model.detokenize(tokenized_prompt[:max(0, context_length - self.max_length)]), 'utf-8')

        return prompt

    def invoke(self, *args, **kwargs):
        """Generates text from the model."""
        output = []
        stream = kwargs.pop("stream", False)

        generated_texts = []
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            model_input_kwargs = {key: kwargs[key] for key in [
                "suffix", "max_tokens", "temperature", "top_p", "logprobs", "echo", "repeat_penalty", "top_k", "stop"] if key in kwargs}

        if stream:
            for token in self.model(prompt, stream=True, **model_input_kwargs):
                generated_texts.append(token['choices'][0]['text'])
        else:
            output = self.model(prompt, **model_input_kwargs)
            generated_texts = [o['text'] for o in output['choices']]
        return generated_texts

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """Check if the model is supported."""
        return bool(model_name_or_path)
