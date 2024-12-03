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
from requests.exceptions import Timeout

# import anthropic
# import langfun as lf
import openai
import pyglove as pg

# pylint: disable=g-bad-import-order
from safe_common import modeling_utils
from safe_common import shared_config
from safe_common import utils
from openai import OpenAI
import concurrent.futures
# pylint: enable=g-bad-import-order
import httpx
def timeout_handler():
  print("Function exceeded the maximum allowed time and will be terminated.")
  # raise TimeoutError("Function execution exceeded the limit.")

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
    self.client = OpenAI(base_url="http://172.16.34.29:5999/v1")
    self.model_name = "gpt-4o-mini"


  def run_with_timeout(self, messages, temp, timeout):
    # Set up a timer
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()  # Start the timer
    try:
      completion = self.client.chat.completions.create(
          model=self.model_name,
          temperature=temp,
          messages=messages)
      response = completion.choices[0].message.content
      return response
    finally:
      timer.cancel()  # Cancel the timer if the function returns before the timeout
    return None

  def generate(
      self,
      prompt: str,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 1,
      timeout: int = 20,
      retry_interval: int = 1,
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
        if temperature is not None:
            temp = temperature
        else:
            temp = self.temperature
        # completion = self.client.chat.completions.create(
        #     model=self.model_name,
        #     temperature=temp,
        #     messages=messages)
        # response = completion.choices[0].message.content
        response = self.run_with_timeout(messages, temp, timeout)
        num_attempts += 1
        return response
      except Exception as e:
        utils.maybe_print_error(e)
        # time.sleep(retry_interval)

    # return response

  def print_config(self) -> None:
    settings = {
        'model_name': self.model_name,
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'show_responses': self.show_responses,
        'show_prompts': self.show_prompts,
    }
    print(utils.to_readable_json(settings))



