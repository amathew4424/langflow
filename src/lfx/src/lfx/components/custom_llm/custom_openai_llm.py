from typing import Any

from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput, DictInput, IntInput, SecretStrInput, SliderInput, StrInput
from lfx.log.logger import logger


class CustomOpenAICompatibleComponent(LCModelComponent):
    display_name = "Custom OpenAI-Compatible"
    description = "Generates text using any OpenAI-compatible API server (e.g., on-prem, self-hosted, or third-party)."
    icon = "Bot"
    name = "CustomOpenAICompatibleModel"

    inputs = [
        *LCModelComponent.get_base_inputs(),
        StrInput(
            name="base_url",
            display_name="Base URL",
            advanced=False,
            info="The base URL of the OpenAI-compatible API server (e.g., http://localhost:8000/v1).",
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            info="The name of the model to use on the server.",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="The API key for authentication (optional for local servers).",
            advanced=False,
            value="",
            required=False,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            show=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="Controls the reproducibility of the job. Set to -1 to disable.",
            advanced=True,
            value=-1,
            required=False,
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="The maximum number of retries to make when generating.",
            advanced=True,
            value=3,
            required=False,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="Timeout for requests to the API server.",
            advanced=True,
            value=600,
            required=False,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        logger.debug(f"Executing request with custom OpenAI-compatible model: {self.model_name}")
        api_key_value = None
        if self.api_key:
            if isinstance(self.api_key, SecretStr):
                api_key_value = self.api_key.get_secret_value()
            else:
                api_key_value = str(self.api_key)

        model_kwargs = self.model_kwargs or {}
        if "api_key" in model_kwargs:
            model_kwargs = dict(model_kwargs)
            del model_kwargs["api_key"]

        parameters = {
            "api_key": api_key_value or "no-key",
            "model_name": self.model_name,
            "max_tokens": self.max_tokens or None,
            "model_kwargs": model_kwargs,
            "base_url": self.base_url,
            "temperature": self.temperature if self.temperature is not None else 0.1,
            "max_retries": self.max_retries if self.max_retries is not None else 3,
            "timeout": self.timeout if self.timeout is not None else 600,
        }

        if self.seed is not None and self.seed != -1:
            parameters["seed"] = self.seed

        output = ChatOpenAI(**parameters)
        if self.json_mode:
            output = output.bind(response_format={"type": "json_object"})

        return output

    def _get_exception_message(self, e: Exception):
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:  # noqa: ARG002
        return build_config
