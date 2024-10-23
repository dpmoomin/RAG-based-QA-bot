from openai import OpenAI

class OpenAILanguageModel:
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.125, max_tokens=500):
        """
        초기화 메서드입니다.

        Parameters:
            api_key (str): OpenAI API 키.
            model (str): 사용할 모델 이름 (기본값: gpt-3.5-turbo).
            temperature (float): 생성된 텍스트의 다양성을 제어하는 매개변수 (기본값: 0.4).
            max_tokens (int): 생성할 최대 토큰 수 (기본값: 500).
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, messages):
        """
        주어진 메시지 목록을 사용하여 응답을 생성합니다.

        Parameters:
            messages (list): 대화 메시지 목록. 각 메시지는 role (system, user, assistant)과 content (메시지 내용)을 포함하는 사전입니다.

        Returns:
            str: 생성된 텍스트 응답. 예외가 발생하면 에러 메시지를 반환합니다.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # 최신 패키지에서 응답 읽기
            return response.choices[0].message.content.strip()

        except Exception as e:
            # 기타 예외 처리
            print(f"An error occurred: {e}")
            return "알 수 없는 오류가 발생했습니다."
