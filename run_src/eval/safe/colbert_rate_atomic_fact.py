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
"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
import time
from typing import Any

# pylint: disable=g-bad-import-order
from safe_common import open_modeling
from safe_common import shared_config
from safe_common import utils
from eval.safe import config as safe_config
from eval.safe import query_serper
# pylint: enable=g-bad-import-order
# from langchain_community.tools import DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()
import requests
import threading
import json
import time
SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_CONTEXT_PLACEHOLDER = '[CONTEXT]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT, a CONTEXT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT in the given CONTEXT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

CONTEXT:
{_CONTEXT_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT, a CONTEXT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given CONTEXT, you can use the given KNOWLEDGE to support your decision if necessary. \
The STATEMENT is supported if it is a proper action or reasoning given the CONTEXT.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. 
4. If the STATEMENT is supported by the CONTEXT, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT. 
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets. 
Remember to only choose "{SUPPORTED_LABEL}" if you are 100% sure with the STATEMENT given the CONTEXT and KNOWLEDGE.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

CONTEXT:
{_CONTEXT_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

@dataclasses.dataclass()
class GoogleSearchResult:
  query: str
  result: str


@dataclasses.dataclass()
class FinalAnswer:
  response: str
  answer: str


def call_search(
    search_query: str,
    search_type: str = safe_config.search_type,
    num_searches: int = safe_config.num_searches,
    serper_api_key: str = shared_config.serper_api_key,
    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
) -> str:
  """Call Google Search to get the search result."""
  global rank_proxies
  if search_type == 'serper':
    serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
    return serper_searcher.run(search_query, k=num_searches)
  elif search_type == 'medrag':
    query = "http://172.16.34.21:6000/api/search?query="+search_query+"&k="+str(num_searches)
    x = requests.get(query)
    jsobj = json.loads(x.text)
    str_res = ""
    for idx in range(len(jsobj['topk'])):
      str_res = str_res + jsobj['topk'][idx]['text'] +"\n"
    return str_res
  else:
    raise ValueError(f'Unsupported search type: {search_type}')

def timeout_handler():
  print("Function exceeded the maximum allowed time and will be terminated.")
  raise TimeoutError("Function execution exceeded the limit.")

def run_with_timeout(func, query, timeout):
  # Set up a timer
  timer = threading.Timer(timeout, timeout_handler)
  timer.start()  # Start the timer
  try:
    result = func(query)
    # ser_result = call_search(query, search_type='serper')
    # import ipdb; ipdb.set_trace()
  finally:
    timer.cancel()  # Cancel the timer if the function returns before the timeout
  return result


def maybe_get_next_search(
    atomic_fact: str,
    context: str,
    past_searches: list[GoogleSearchResult],
    model: open_modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> GoogleSearchResult | None:
  """Get the next query from the model."""
  knowledge = '\n'.join([s.result for s in past_searches])
  knowledge = 'N/A' if not knowledge else knowledge
  full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = full_prompt.replace(_CONTEXT_PLACEHOLDER, context)
  full_prompt = utils.strip_string(full_prompt)
  model_response = model.generate(full_prompt, do_debug=debug, temperature=0.1)
  query = utils.extract_first_code_block(model_response, ignore_language=True)
  if model_response and query:
    # print("query: ", query)
    # Usage
    # try:
    #   result = run_with_timeout(call_search, query, 180)  # Set timeout to 3 seconds
    # except TimeoutError as e:
    #   print(e)
    #   return None
    result = call_search(query)
    # ser_result = call_search(query, search_type='serper')
    # import ipdb; ipdb.set_trace()
    return GoogleSearchResult(query=query, result=result)
  return None


def maybe_get_final_answer(
    atomic_fact: str,
    context: str,
    searches: list[GoogleSearchResult],
    model: open_modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> FinalAnswer | None:
  """Get the final answer from the model."""
  knowledge = '\n'.join([search.result for search in searches])
  full_prompt = _FINAL_ANSWER_FORMAT.replace(
      _STATEMENT_PLACEHOLDER, atomic_fact
  )
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = full_prompt.replace(_CONTEXT_PLACEHOLDER, context)
  full_prompt = utils.strip_string(full_prompt)
  model_response = model.generate(full_prompt, do_debug=debug, temperature=0)
  answer = utils.extract_first_square_brackets(model_response)
  answer = re.sub(r'[^\w\s]', '', answer).strip()
  # print(full_prompt)
  # print(model_response)
  # import ipdb; ipdb.set_trace()

  if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
    return FinalAnswer(response=model_response, answer=answer)

  return None


def check_atomic_fact(
    atomic_fact: str,
    context: str,
    rater: open_modeling.Model,
    max_steps: int = safe_config.max_steps,
    max_retries: int = safe_config.max_retries,
    debug: bool = safe_config.debug_safe,
) -> tuple[FinalAnswer | None, dict[str, Any]]:
  """Check if the given atomic fact is supported."""
  search_results = []

  for idx in range(max_steps):
    next_search, num_tries = None, 0

    while not next_search and num_tries <= max_retries:
      next_search = maybe_get_next_search(atomic_fact, context, search_results, rater)
      # import ipdb; ipdb.set_trace()
      num_tries += 1

    if next_search is None:
      utils.maybe_print_error('Unsuccessful parsing for `next_search`')
      continue
    else:
      search_results.append(next_search)

  search_dicts = {
      'google_searches': [dataclasses.asdict(s) for s in search_results]
  }
  final_answer, num_tries = None, 0

  while not final_answer and num_tries <= max_retries:
    num_tries += 1
    final_answer = maybe_get_final_answer(
        atomic_fact, context, searches=search_results, model=rater, debug=debug
    )

  if final_answer is None:
    utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

  return final_answer, search_dicts


# search_query += f' {search_postamble}' if search_postamble else ''
#     while True:
#       try:
#         result = search.run(search_query)
#         if isinstance(result, str):
#           return result
#       except Exception as e:
#         print(e)
#         time.sleep(120)

# while True:
    #   try:
    #     ddgs = DDGS(proxy="http://x1UGGwifXXEvbvCL:iqfYSFmkaVKhdvYD_streaming-1@geo.iproyal.com:12321", timeout=20)
    #     results = ddgs.text(search_query, max_results=3)
    #     str_res = ""
    #     for res in results:
    #       str_res = str_res + res['body']
    #     return str_res
    #   except Exception as e:
    #     print(e)