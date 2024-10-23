from openai import OpenAI

class OpenAILanguageModel:
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.125, max_tokens=500):
        """
        OpenAI 언어 모델 초기화.

        Parameters:
            api_key (str): OpenAI API 키
            model (str): 사용할 모델 이름 (기본값: "gpt-3.5-turbo")
            temperature (float): 생성 텍스트의 다양성 (기본값: 0.125)
            max_tokens (int): 생성할 최대 토큰 수 (기본값: 500)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, messages):
        """
        메시지 목록을 기반으로 응답을 생성합니다.

        Parameters:
            messages (list): 대화 메시지 목록

        Returns:
            str: 생성된 응답 텍스트
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            return "알 수 없는 오류가 발생했습니다."
