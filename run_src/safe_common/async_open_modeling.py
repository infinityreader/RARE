# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sets up language models to be used."""

from concurrent import futures
import functools
import logging
import os
import threading
import time
from typing import Any, Annotated, Optional

# import anthropic
# import langfun as lf
import openai
import pyglove as pg

# pylint: disable=g-bad-import-order
from common import modeling_utils
from common import shared_config
from common import utils
from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union
# pylint: enable=g-bad-import-order
import asyncio

async def call_vllm_batch_api(
        client: AsyncOpenAI,
        prompts: List[str],
) -> List[str]:
    responses = await client.completions.create(
        model="qwen72b_with_openai_api",
        prompt=prompts
    )
    assert len(responses.choices) == len(prompts)  # n=1
    responses_list = [responses.choices[i].text for i in range(len(prompts))]
    return responses_list

class Model:
  """Class for storing any single language model."""

  def __init__(
      self,
      model_name: str,
      temperature: float = 0.5,
      max_tokens: int = 2048,
      show_responses: bool = False,
      show_prompts: bool = False,
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.show_responses = show_responses
    self.show_prompts = show_prompts
    # self.clientnvidia = OpenAI(base_url="http://0.0.0.0:7999/v1", api_key="original")
    self.client = AsyncOpenAI(base_url="http://10.10.0.172:7999/v1", api_key="original")
    self.tokenizer = AutoTokenizer.from_pretrained(
        "/home/data/30_LLaMa_model_weights_HF/Qwen2-72B-Instruct-AWQ",
        trust_remote_code=True,
    )
    if "qwen" in self.tokenizer.__class__.__name__.lower():
        self.tokenizer.bos_token = "<|im_start|>"
        # tokenizer.bos_token_id = 151644
    if self.tokenizer.pad_token is None:
        if self.tokenizer.unk_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token

  def generate(
      self,
      prompt: str,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 1000,
      timeout: int = 60,
      retry_interval: int = 10,
  ) -> str:
    """Generates a response to a prompt."""
    prompt = modeling_utils.add_format(prompt, '', self.model_name)
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens
    response, num_attempts = '', 0


    while not response and num_attempts < max_attempts:
      try:
        messages = [{"role": "system", "content": "You are a helpful chatbot, please chat with me."},
                    {"role": "user", "content": prompt}]
        prompt_sample_list = [
            self.tokenizer.apply_chat_template(
                [messages[0], {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,  # chat template
                skip_special_tokens=False,
            )
        ]
        answer_list = asyncio.run(call_vllm_batch_api(client=self.client, prompts=prompt_sample_list))
      except Exception as e:
        utils.maybe_print_error(e)
        time.sleep(retry_interval)
        num_attempts += 1

    return answer_list[0]

  def print_config(self) -> None:
    settings = {
        'model_name': self.model_name,
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'show_responses': self.show_responses,
        'show_prompts': self.show_prompts,
    }
    print(utils.to_readable_json(settings))



