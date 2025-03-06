import instructor
import openai
from prompts.utils import Response

from .llm_model import LLMModel, LLMModelType


class LlamaFactoryVllm(LLMModel):
    def __init__(self, model_name: str, use_instructor: bool):
        super().__init__(model_name, LLMModelType.LLAMA_FACTORY_VLLM, use_instructor)
        if self.use_instructor:
            raise NotImplementedError("Instructor not implemented for LlamaFactoryVllm")
        self.client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")

    def get_model_response(self, messages: list[dict]) -> Response:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )
            content = completion.choices[0].message.content
            response = self.get_response_from_completion(content)
        except Exception as e:
            import pdb

            pdb.set_trace()

        return response
