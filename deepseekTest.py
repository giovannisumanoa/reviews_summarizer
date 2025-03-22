from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
API_KEY = "sk-b49e0cc821cc41cea896e5fa21322c90"

from openai import OpenAI

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you today?"},
    ],
    stream=False
)

print(response.choices[0].message.content)