from prompts.prompt_templates import DEFAULT_SYSTEM_PROMPT
from models.language_model import OpenAILanguageModel
from config.settings import OPENAI_API_KEY

class RetrievalQAChain:
    def __init__(self, retriever, prompt_template=DEFAULT_SYSTEM_PROMPT, language_model=None):
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.language_model = language_model or OpenAILanguageModel(api_key=OPENAI_API_KEY)

    def run(self, query):
        # 문서 검색
        docs = self.retriever.retrieve(query)
        context = "\n\n".join(docs)

        # 프롬프트 생성
        system_prompt = self.prompt_template.format(context=context)

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 답변 생성
        answer = self.language_model.generate(messages)
        return answer
