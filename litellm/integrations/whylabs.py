import backoff
import json
import os
import requests
from typing import Any, Dict, Literal, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.utils import (
    get_formatted_prompt,
    get_response_string,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextCompletionResponse,
)
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

try:
    from icecream import ic
except:
    def ic(x):
        print(x)
        return x


def _ic(fn):
    ic(fn)


MAX_REQUEST_TIME = 60  # seconds
MAX_REQUEST_TRIES = 10


def get_output_str_from_response(response_obj, kwargs):
    if response_obj is None:
        return None
    output = None
    if kwargs.get("call_type", None) == "embedding" or isinstance(response_obj, EmbeddingResponse):
        output = None
    elif isinstance(response_obj, ModelResponse):
        output = get_response_string(response_obj)
    elif isinstance(response_obj, TextCompletionResponse):
        output = response_obj.choices[0].text

    return output


class WhyLabsBase:
    def __init__(self, **kwargs):
        self._endpoint = os.environ.get("WHYLOGS_API_ENDPOINT")
        self._api_key = os.environ.get("WHYLOGS_API_KEY")
        self._dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
        self._org_id = os.environ.get("WHYLABS_DEFAULT_ORG_ID")
        self._route = ""

    @backoff.on_predicate(
        backoff.expo,
        lambda x: x.status_code != 200,
        max_time=MAX_REQUEST_TIME,
        max_tries=MAX_REQUEST_TRIES,
        jitter=backoff.full_jitter,
    )
    def _do_request(self, data: Dict[str, str]) -> requests.Response:
        headers = {"X-API-Key": self._api_key}
        r = ic(requests.post(f"{self._endpoint}/{self._route}", json=data, headers=headers))
        return r

        
class WhyLabsLogger(CustomLogger, WhyLabsBase):
    def __init__(self, **kwargs):
        CustomLogger.__init__(self, **kwargs)
        WhyLabsBase.__init__(self, **kwargs)
        self._route = "log/llm"

    def _log_event(self, kwargs, response_obj, start_time, end_time):
        #ic(kwargs)
        call_type = kwargs.get("call_type", "litellm")
        ic(call_type)
        call_type = "completion" if call_type == "acompletion" else call_type  # Hmmm... LiteLLM bug?
        ic(call_type)
        prompt = get_formatted_prompt(data=kwargs, call_type=call_type)
        ic(prompt)
        response = get_response_string(response_obj)
        ic(response)

        data = {
            "prompt": prompt,
            "response": response,
            "datasetId": self._dataset_id,
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


class WhyLabsGuardrail(CustomGuardrail, WhyLabsBase):
    def __init__(self, **kwargs):
        CustomGuardrail.__init__(self, **kwargs)
        WhyLabsBase.__init__(self, **kwargs)
        self._route = "evaluate"

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        verbose_proxy_logger.debug("****************************  begin async_pre_call_hook *******************************")
        #verbose_proxy_logger.debug(ic.format(user_api_key_dict))
        #verbose_proxy_logger.debug(ic.format(data))
        #verbose_proxy_logger.debug(ic.format(call_type))

        blocked = False
        block_messages: List[str] = []
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    request = {
                        "prompt": _content,
                        "datasetId": self._dataset_id,
                    }
                    response = ic(self._do_request(request).json())
                    action, action_message = response["action"]["action_type"], response["action"]["message"]
                    ic(prompt)
                    ic(response)
                    # If the response included a modified prompt, we would replace the message in data
                    # and pass it on to the next guardrail or LLM: message["content"] = <modified prompt>

                    if action == "block":
                        blocked = True
                        block_messages.append(action_message)
                        verbose_prox_logger.debug(f"Prompt blocked by WhyLabs guardrail: {action_message}")
                    elif action == "flag":
                        verbose_proxy_logger.debug(f"Prompt flagged by WhyLabs guardrail: {action_message}")
                    elif action == "pass":
                        verbose_proxy_logger.debug("Prompt passed by WhyLabs guardrail")
                    else:
                        verbose_proxy_logger.warning(f"WhyLabs guardrail returned unknown action: {action} {action_message}")

        if blocked:
            raise ValueError(f"Prompt(s) blocked by WhyLabs guardrail: {';'.join(block_messages)})")
        verbose_proxy_logger.debug("****************************  end async_pre_call_hook *******************************")
        verbose_proxy_logger.debug(ic.format(data))
        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Any:
        verbose_proxy_logger.debug("****************************  begin async_moderation_hook *******************************")
        #verbose_proxy_logger.debug(ic.format(user_api_key_dict))
        #verbose_proxy_logger.debug(ic.format(data))
        #verbose_proxy_logger.debug(ic.format(call_type))

        blocked = False
        block_messages: List[str] = []
        _messages = data.get("messages")
        if _messages:
            for message in _messages:  # should these be concatenated so we can hit whylogs just once?
                _content = message.get("content")
                if isinstance(_content, str):
                    data = {
                        "prompt": _content,
                        #"id": self._dataset_id,
                        "datasetId": self._dataset_id,
                    }
                    response = ic(self._do_request(data).json())
                    action, action_message = response["action"]["action_type"], response["action"]["message"]

                    # The prompt has already be sent on the LLM, so we can't modify it. We can only block it.

                    if action == "block":
                        blocked = True
                        block_messages.append(action_message)
                        verbose_prox_logger.debug(f"Prompt blocked by WhyLabs guardrail: {action_message}")
                    elif action == "flag":
                        verbose_proxy_logger.debug(f"Prompt flagged by WhyLabs guardrail: {action_message}")
                    elif action == "pass":
                        verbose_proxy_logger.debug("Prompt passed by WhyLabs guardrail")
                    else:
                        verbose_proxy_logger.warning(f"WhyLabs guardrail returned unknown action: {action} {action_message}")

        if blocked:
            raise ValueError(f"Prompt(s) blocked by WhyLabs guardrail: {';'.join(block_messages)})")
        verbose_proxy_logger.debug("****************************  end async_moderation_hook *******************************")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        verbose_proxy_logger.debug("****************************  begin async_post_call_hook *******************************")
        #verbose_proxy_logger.debug(ic.format(user_api_key_dict))
        verbose_proxy_logger.debug(ic.format(data))
        verbose_proxy_logger.debug(ic.format(response))

        verbose_proxy_logger.debug(ic.format(data.get("messages")))
        answer = get_response_string(response)
        verbose_proxy_logger.debug(ic.format(answer))

        verbose_proxy_logger.debug("****************************  end async_post_call_hook *******************************")
