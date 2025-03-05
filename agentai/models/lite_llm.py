import instructor
import litellm
from prompts import Response

from .llm_model import LLMModel, LLMModelType


class LiteLLM(LLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, LLMModelType.LiteLLM)

        self.client = instructor.from_litellm(litellm.completion)

    def get_model_response(self, messages: list[dict]) -> Response:
        response: Response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=Response,
        )

        return response
