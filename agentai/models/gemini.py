import os

import google.generativeai as genai
import instructor
from prompts.utils import Response

from .llm_model import LLMModel, LLMModelType


class Gemini(LLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, LLMModelType.GEMINI)

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=self.model_name),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def get_model_response(self, messages: list[dict]) -> Response:
        response: Response = self.client.create(
            response_model=Response,
            messages=messages,
        )

        return response
