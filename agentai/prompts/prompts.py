from .prompts_basic import (
    click_example_response,
    format_instruction,
    input_text_example_response,
    instruction_prompt,
    next_action,
)
from .utils import PromptKeys, Response

system_prompt = {
    PromptKeys.INSTRUCTION: instruction_prompt,
    PromptKeys.RESPONSE_JSON_DESCRIPTION: Response.model_json_schema(),
    PromptKeys.FORMAT_INSTRUCTION: format_instruction,
    PromptKeys.EXAMPLE_RESPONSE: [
        click_example_response.model_dump(),
        input_text_example_response.model_dump(),
    ],
}

user_prompt = {
    PromptKeys.NEXT_STEP: next_action,
}
