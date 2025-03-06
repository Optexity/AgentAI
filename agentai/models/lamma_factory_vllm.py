import instructor
import openai
from prompts.utils import Response

from .llm_model import LLMModel, LLMModelType


class LlamaFactoryVllm(LLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name, LLMModelType.LLAMA_FACTORY_VLLM)
        self.client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")

    def get_model_response(self, messages: list[dict]) -> Response:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )
            response = Response.model_validate_json(
                completion.choices[0].message.content
            )
        except Exception as e:
            import pdb

            pdb.set_trace()

        return response
