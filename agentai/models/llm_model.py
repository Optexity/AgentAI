import ast
import re
from enum import Enum

from prompts.utils import Response


class LLMModelType(Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    AZURE = "azure"
    CUSTOM = "custom"
    LiteLLM = "LiteLLM"
    LLAMA_FACTORY_VLLM = "llama_factory_vllm"


class GeminiModels:
    GEMINI_2_0_FLASH = "gemini-2.0-flash"


class VLLMModels:
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class LLMModel:
    def __init__(self, model_name: str, model_type: LLMModelType, use_instructor: bool):
        self.model_type = model_type
        self.model_name = model_name
        self.use_instructor = use_instructor

    def get_model_response(self, messages: list[dict]) -> Response:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_response_from_completion(self, content: str) -> Response:
        pattern = r"```json\n(.*?)\n```"
        json_blocks = re.findall(pattern, content, re.DOTALL)
        for block in json_blocks:
            block = block.strip()
            try:
                response = Response.model_validate_json(block)
                return response
            except Exception as e:
                try:
                    block_dict = ast.literal_eval(block)
                    response = Response.model_validate(block_dict)
                    return response
                except Exception as e:
                    continue
        raise ValueError("Could not parse response from completion.")
