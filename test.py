import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)