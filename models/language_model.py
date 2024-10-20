import openai

class OpenAILanguageModel:
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.7, max_tokens=500):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = self.api_key

    def generate(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message['content'].strip()
