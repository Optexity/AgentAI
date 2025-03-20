import os

import google.generativeai as genai
import instructor
import litellm
from agentai.prompts.utils import Response

from .llm_model import GeminiModels, LLMModel


class Gemini(LLMModel):
    def __init__(self, model_name: GeminiModels, use_instructor: bool):
        super().__init__(model_name, use_instructor)

        if self.use_instructor:
            self.model_name = f"models/{model_name.value}"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.client = instructor.from_gemini(
                client=genai.GenerativeModel(model_name=self.model_name),
                mode=instructor.Mode.GEMINI_JSON,
            )
        else:
            self.model_name = f"gemini/{model_name.value}"

    def get_model_response(self, messages: list[dict]) -> Response:
        if self.use_instructor:
            response: Response = self.client.create(
                response_model=Response, messages=messages
            )
        else:
            try:
                completion = litellm.completion(
                    model=self.model_name, messages=messages
                )
                content = completion.choices[0].message.content
                response = self.get_response_from_completion(content)

            except Exception as e:
                print("Super exception:", e)

        return response
