import ast
import re
from enum import Enum, unique

from agentai.prompts.utils import Response


@unique
class GeminiModels(Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    TUNED_MODELS_HUBSPOT_V1 = "hubspotv1-pga2a8p0hgrq"


@unique
class VLLMModels(Enum):
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class LLMModel:
    def __init__(self, model_name: GeminiModels | VLLMModels, use_instructor: bool):

        self.model_name = model_name
        self.use_instructor = use_instructor

    def get_model_response(self, messages: list[dict]) -> Response:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def extract_json_objects(self, text):
        stack = []  # Stack to track `{` positions
        json_candidates = []  # Potential JSON substrings

        # Iterate through the text to find balanced { }
        for i, char in enumerate(text):
            if char == "{":
                stack.append(i)  # Store index of '{'
            elif char == "}" and stack:
                start = stack.pop()  # Get the last unmatched '{'
                json_candidates.append(text[start : i + 1])  # Extract substring

        return json_candidates

    def get_response_from_completion(self, content: str) -> Response:
        patterns = [r"```json\n(.*?)\n```"]
        json_blocks = []
        for pattern in patterns:
            json_blocks += re.findall(pattern, content, re.DOTALL)
        json_blocks += self.extract_json_objects(content)
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
