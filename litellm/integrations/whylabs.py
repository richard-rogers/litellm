import os

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.utils import get_formatted_prompt, get_response_string

from whylogs_container_client import AuthenticatedClient
import whylogs_container_client.api.llm.log_llm as LogLlm
import whylogs_container_client.api.llm.evaluate as Evaluate
from whylogs_container_client import Client

from whylogs_container_client.models.evaluation_result import EvaluationResult
from whylogs_container_client.models import LogRequest
from whylogs_container_client.models.llm_validate_request import LLMValidateRequest
from whylogs_container_client.models.validation_result import ValidationResult

from icecream import ic


def _ic(fn):
    ic(fn)


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
        whylogs_endpoint = os.environ.get("WHYLOGS_API_ENDPOINT")
        whylogs_api_key = os.environ.get("WHYLOGS_API_KEY")
        self._client = AuthenticatedClient(
            base_url=whylogs_endpoint,
            token=whylogs_api_key,
            prefix="",
            auth_header_name="X-API-Key"
        )
        self._dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
        self._org_id = os.environ.get("WHYLABS_DEFAULT_ORG_ID")

    def _log_event(self, kwargs, response_obj, start_time, end_time):
        #ic(kwargs)
        call_type = kwargs.get("call_type", "litellm")
        ic(call_type)
        prompt = get_formatted_prompt(data=kwargs, call_type=call_type)
        ic(prompt)
        response = get_response_string(response_obj)
        ic(response)

        request = LLMValidateRequest(
            prompt=prompt,
            response=response,
            dataset_id=self._dataset_id,
            id=self._dataset_id,
        )
        ic(LogLlm.sync_detailed(client=self._client, body=request))

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
