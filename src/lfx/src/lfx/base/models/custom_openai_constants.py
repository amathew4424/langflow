from .model_metadata import create_model_metadata

# Placeholder model for the Custom OpenAI-Compatible provider.
# Since the actual models depend on the user's server, this provides
# a default entry so the provider appears in the unified model selector.
# Users should type their actual model name in the combobox.
CUSTOM_OPENAI_MODELS_DETAILED = [
    create_model_metadata(
        provider="Custom OpenAI-Compatible",
        name="custom-model",
        icon="Bot",
        tool_calling=True,
        default=True,
    ),
]
