from pydantic.v1 import SecretStr

from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput, DictInput, IntInput, SecretStrInput, SliderInput, StrInput


class CustomLLMComponent(LCModelComponent):
    display_name = "Custom LLM"
    description = "Connect to any OpenAI-compatible language model API by providing a base URL, API key, and model name."
    icon = "bot"
    name = "CustomLLM"

    inputs = [
        *LCModelComponent.get_base_inputs(),
        StrInput(
            name="base_url",
            display_name="Base URL",
            info="The base URL of the API (e.g. https://api.openai.com/v1).",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="The API key for authentication.",
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            info="The name of the model to use (e.g. gpt-4o, llama-3-70b).",
            required=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="The maximum number of tokens to generate.",
            advanced=True,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            value=700,
            advanced=True,
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            info="If True, the model will output JSON.",
            advanced=True,
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            info="Additional keyword arguments to pass to the model.",
            advanced=True,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        from langchain_openai import ChatOpenAI

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
            "api_key": api_key_value,
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature if self.temperature is not None else 0.1,
            "max_tokens": self.max_tokens or None,
            "model_kwargs": model_kwargs,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

        if self.seed:
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
            if e.body is not None:
                message = e.body.get("message")
                if message:
                    return message
            return str(e)
        return None
