import instructor
from openai import OpenAI
from prompts.utils import Response

from .llm_model import LLMModel, LLMModelType


class LlamaFactoryVllm(LLMModel):
    def __init__(self, model_name: str, use_instructor: bool, port: int):
        super().__init__(model_name, LLMModelType.LLAMA_FACTORY_VLLM, use_instructor)
        assert port is not None, "Port must be provided for LlamaFactoryVllm"
        if self.use_instructor:
            raise NotImplementedError("Instructor not implemented for LlamaFactoryVllm")
        self.client = OpenAI(base_url=f"http://0.0.0.0:{port}/v1", api_key="dummy")

    def get_model_response(self, messages: list[dict]) -> Response:

        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        content = completion.choices[0].message.content
        response = self.get_response_from_completion(content)

        return response
