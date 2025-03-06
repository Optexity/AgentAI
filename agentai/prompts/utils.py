from pydantic import BaseModel, Field


class Roles:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PromptKeys:
    INSTRUCTION = "instruction"
    RESPONSE_JSON_DESCRIPTION = "output_description"
    FORMAT_INSTRUCTION = "format_instruction"
    EXAMPLE_RESPONSE = "example_response"
    TRAJECTORIES = "trajectories"
    AVAILABLE_ACTIONS = "available_actions"
    PREVIOUS_ACTION_ERROR = "previous_action_error"
    PREVIOUS_ACTION = "previous_action"
    CURRENT_OBSERVATION = "current_observation"
    NEXT_STEP = "next_step"
    GOAL = "goal"


class PromptStyle:
    BEGIN = "begin"
    END = "end"
    DESCRIPTION = "description"
    LIST_SEPARATOR = "list_separator"


class Response(BaseModel):
    """
    The response format for the action to take. Think step-by-step through the action you want to take.
    """

    # observation: str = Field(
    #     description="Summary of the observation you are responding to. This is the current state of the page."
    # )
    # reasoning: str = Field(description="Your reasoning for taking this action.")
    action_name: str = Field(
        description="The action_name should be one of the available actions"
    )
    action_params: dict = Field(
        description="""The parameters of the action you want to take. Must be valid JSON. 
        The action_params should be valid for that action.
        The action_params should be a dictionary with the parameters of the action.
            {
            "param1": "value1",
            "param2": "value2"
        }
        """
    )


style = {
    PromptKeys.INSTRUCTION: {
        PromptStyle.BEGIN: "[INSTRUCTION]",
        PromptStyle.END: "[/INSTRUCTION]",
    },
    PromptKeys.RESPONSE_JSON_DESCRIPTION: {
        PromptStyle.BEGIN: "[RESPONSE_JSON_DESCRIPTION]",
        PromptStyle.END: "[/RESPONSE_JSON_DESCRIPTION]",
    },
    PromptKeys.FORMAT_INSTRUCTION: {
        PromptStyle.BEGIN: "[FORMAT_INSTRUCTION]",
        PromptStyle.END: "[/FORMAT_INSTRUCTION]",
    },
    PromptKeys.EXAMPLE_RESPONSE: {
        PromptStyle.BEGIN: "[EXAMPLE_RESPONSE]",
        PromptStyle.END: "[/EXAMPLE_RESPONSE]",
        PromptStyle.LIST_SEPARATOR: "Example",
    },
    "trajectories": {
        PromptStyle.BEGIN: "[TRAJECTORIES]",
        PromptStyle.END: "[/TRAJECTORIES]",
    },
    PromptKeys.AVAILABLE_ACTIONS: {
        PromptStyle.BEGIN: "[AVAILABLE_ACTIONS]",
        PromptStyle.END: "[/AVAILABLE_ACTIONS]",
        PromptStyle.LIST_SEPARATOR: "Action",
    },
    PromptKeys.PREVIOUS_ACTION_ERROR: {
        PromptStyle.BEGIN: "[PREVIOUS_ACTION_ERROR]",
        PromptStyle.END: "[/PREVIOUS_ACTION_ERROR]",
    },
    "previous_action": {
        PromptStyle.BEGIN: "[PREVIOUS_ACTION]",
        PromptStyle.END: "[/PREVIOUS_ACTION]",
    },
    PromptKeys.CURRENT_OBSERVATION: {
        PromptStyle.BEGIN: "[CURRENT_OBSERVATION]",
        PromptStyle.END: "[/CURRENT_OBSERVATION]",
        PromptStyle.DESCRIPTION: "Current AXTree of the webpage. Use the number on the items to extract bid.",
    },
    PromptKeys.NEXT_STEP: {
        PromptStyle.BEGIN: "[NEXT_STEP]",
        PromptStyle.END: "[/NEXT_STEP]",
    },
    PromptKeys.GOAL: {PromptStyle.BEGIN: "[GOAL]", PromptStyle.END: "[/GOAL]"},
}
