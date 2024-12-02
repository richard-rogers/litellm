import backoff
import json
import os
import requests
from typing import Dict

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.utils import get_formatted_prompt, get_response_string

try:
    from icecream import ic
except:
    def ic(x):
        print(x)


def _ic(fn):
    ic(fn)


MAX_REQUEST_TIME = 60  # seconds
MAX_REQUEST_TRIES = 10


def get_output_str_from_response(response_obj, kwargs):
    if response_obj is None:
        return None
    output = None
    if kwargs.get("call_type", None) == "embedding" or isinstance(response_obj, litellm.EmbeddingResponse):
        output = None
    elif isinstance(response_obj, litellm.ModelResponse):
        output = get_response_string(response_obj)
    elif isinstance(response_obj, litellm.TextCompletionResponse):
        output = response_obj.choices[0].text

    return output


class WhyLabsLogger(CustomLogger):
    def __init__(self):
        self._endpoint = os.environ.get("WHYLOGS_API_ENDPOINT")
        self._api_key = os.environ.get("WHYLOGS_API_KEY")
        self._dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
        self._org_id = os.environ.get("WHYLABS_DEFAULT_ORG_ID")

    @backoff.on_predicate(
        backoff.expo,
        lambda x: x.status_code != 200,
        max_time=MAX_REQUEST_TIME,
        max_tries=MAX_REQUEST_TRIES,
        jitter=backoff.full_jitter,
    )
    def _do_request(self, data: Dict[str, str]) -> requests.Response:
        headers = {"X-API-Key": self._api_key}
        r = requests.post(f"{self._endpoint}/log/llm", json=data, headers=headers)
        ic(r)
        return r

    def _log_event(self, kwargs, response_obj, start_time, end_time):
        #ic(kwargs)
        call_type = kwargs.get("call_type", "litellm")
        ic(call_type)
        prompt = get_formatted_prompt(data=kwargs, call_type=call_type)
        ic(prompt)
        response = get_response_string(response_obj)
        ic(response)

        data = {
            "prompt": prompt,
            "response": response,
            "id": self._dataset_id,
            "datasetId": self._dataset_id,
            "timestamp": 0,
        }
        self._do_request(data)

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        _ic("log_stream_event")
        self._log_event(kwargs, response_obj, start_time, end_time)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _ic("log_success_event")
        self._log_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _ic("log_failure_event")
        self._log_event(kwargs, response_obj, start_time, end_time)

    #### ASYNC ####

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        _ic("async_log_stream_event")
        self._log_event(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _ic("async_log_success_event")
        self._log_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _ic("async_log_failure_event")
        self._log_event(kwargs, response_obj, start_time, end_time)
