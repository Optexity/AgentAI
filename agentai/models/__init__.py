from .llm_model import GeminiModels, VLLMModels


def get_llm_model(
    model_name: GeminiModels | VLLMModels, use_instructor: bool, port: int = None
):
    if isinstance(model_name, GeminiModels):
        from .gemini import Gemini

        return Gemini(model_name, use_instructor)

    if isinstance(model_name, VLLMModels):
        from .lamma_factory_vllm import LlamaFactoryVllm

        return LlamaFactoryVllm(model_name, use_instructor, port)

    raise ValueError(f"Invalid model type: {model_type}")
