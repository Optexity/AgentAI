from .llm_model import GeminiModels, LLMModelType, VLLMModels


def get_llm_model(model_name: str, model_type: LLMModelType, use_instructor: bool):
    if model_type == LLMModelType.GEMINI:
        from .gemini import Gemini

        return Gemini(model_name, use_instructor)
    elif model_type == LLMModelType.LLAMA_FACTORY_VLLM:
        from .lamma_factory_vllm import LlamaFactoryVllm

        return LlamaFactoryVllm(model_name, use_instructor)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
