from openai import OpenAI

class OpenAIEmbedding:
    def __init__(self, api_key, model="text-embedding-3-small"):
        """
        OpenAI 임베딩 모델 초기화.

        Parameters:
            api_key (str): OpenAI API 키
            model (str): 사용할 임베딩 모델 (기본값: "text-embedding-3-small")
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def get_embedding(self, text):
        """
        주어진 텍스트의 임베딩을 생성합니다.

        Parameters:
            text (str): 임베딩할 텍스트

        Returns:
            list: 임베딩 벡터
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"[오류] 임베딩 생성 실패: {e}")
            return []
