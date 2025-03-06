import openai

client = openai.OpenAI(
    api_key="sk-1234",  # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:8000/v1",  # litellm-proxy-base url
)

response = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "what llm are you"}],
)

print(response)

import litellm

response = litellm.completion(
    model="hosted_vllm/facebook/opt-125m",  # pass the vllm model name
    messages=[{"role": "user", "content": "what llm are you"}],
    api_base="http://0.0.0.0:8000/v1",
    temperature=0.2,
    max_tokens=80,
)

print(response)
