from .llm_model import LLMModel, LLMModelType


def get_llm_model(model_name: str, model_type: LLMModelType):
    if model_type == LLMModelType.GEMINI:
        from .gemini import Gemini

        return Gemini(model_name)
    elif model_type == LLMModelType.LiteLLM:
        from .lite_llm import LiteLLM

        return LiteLLM(model_name)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
