import instructor
import litellm
import openai

from .llm_model import LLMModel, LLMModelType


class LiteLLM(LLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, LLMModelType.LiteLLM)
