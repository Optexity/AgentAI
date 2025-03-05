from enum import Enum


class LLMModelType(Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    AZURE = "azure"
    CUSTOM = "custom"
    LiteLLM = "LiteLLM"


class LLMModel:
    def __init__(self, model_name: str, model_type: LLMModelType):
        self.model_type = model_type
        self.model_name = model_name

    def get_model_response(self, messages: list[dict]) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")
