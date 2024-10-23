from prompts.prompt_templates import (
    DEFAULT_SYSTEM_PROMPT,
    CATEGORY_IDENTIFICATION_PROMPT,
    INTENT_UNDERSTANDING_PROMPT
)
from models.language_model import OpenAILanguageModel
from config.settings import OPENAI_API_KEY
import tiktoken

def count_tokens(text, encoding_name='cl100k_base'):
    """
    텍스트의 토큰 수를 계산합니다.

    Parameters:
        text (str): 입력 텍스트
        encoding_name (str): 사용될 인코딩 이름 (기본값: 'cl100k_base')

    Returns:
        int: 토큰 수
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def truncate_history(history, max_tokens=2048, encoding_name='cl100k_base'):
    """
    대화 이력을 최대 토큰 수에 맞게 자릅니다.

    Parameters:
        history (str): 대화 이력
        max_tokens (int): 최대 허용 토큰 수 (기본값: 2048)
        encoding_name (str): 인코딩 이름 (기본값: 'cl100k_base')

    Returns:
        str: 자른 대화 이력
    """
    current_tokens = count_tokens(history, encoding_name)

    if current_tokens > max_tokens:
        messages = history.split("\n")
        truncated_history = ""

        for message in reversed(messages):
            truncated_history = message + "\n" + truncated_history
            current_tokens = count_tokens(truncated_history, encoding_name)
            if current_tokens > max_tokens:
                break
        return truncated_history.strip()

    return history

class RetrievalQAChain:
    def __init__(self, retriever, language_model=None):
        """
        RetrievalQAChain 초기화 메서드.

        Parameters:
            retriever: 문서 검색을 위한 검색기 객체
            language_model: 언어 모델 객체 (기본값: OpenAILanguageModel)
        """
        self.retriever = retriever
        self.category_prompt = CATEGORY_IDENTIFICATION_PROMPT
        self.intent_prompt = INTENT_UNDERSTANDING_PROMPT
        self.answer_prompt = DEFAULT_SYSTEM_PROMPT
        self.language_model = language_model or OpenAILanguageModel(api_key=OPENAI_API_KEY)
        self.conversation_history = []

    def run(self, query):
        """
        사용자 질문에 대한 답변을 생성합니다.

        Parameters:
            query (str): 사용자 질문

        Returns:
            str: 생성된 답변
        """
        # 1단계: 문서 검색
        retrieved_documents = self.retrieve_documents(query, 5)

        # 2단계: 카테고리 식별
        category = self.identify_category(query, retrieved_documents)
        if category == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.':
            return category
        elif ('•' in category or '-' in category) and (category.count('•') > 1 or category.count('-') > 1):
            print("\n카테고리가 불명확합니다. 아래의 옵션 중에서 선택해 주세요:\n")
            print(category)
            category = input("답변: ").strip()

        # 3단계: 질문의 의도 파악
        intent = self.understand_intent(query, category, retrieved_documents)
        if intent == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.':
            return intent
        elif ('•' in intent or '-' in intent) and (intent.count('•') > 1 or intent.count('-') > 1):
            print("\n의도가 불명확합니다. 아래의 옵션 중에서 선택해 주세요:\n")
            print(intent)
            intent = input("답변: ").strip()

        # 4단계: 답변 생성
        answer = self.generate_answer(query, category, intent, retrieved_documents)

        # 대화 이력 업데이트
        self.update_conversation_history(query, answer, retrieved_documents)
        return answer

    def build_context(self, query, category, intent, retrieved_documents):
        """
        답변 생성을 위한 컨텍스트를 구축합니다.

        Parameters:
            query (str): 사용자 질문
            category (str): 식별된 카테고리
            intent (str): 식별된 의도
            retrieved_documents (list): 검색된 문서 리스트

        Returns:
            str: 구축된 컨텍스트
        """
        history = "\n".join(set(self.conversation_history))
        retrieved_text = "\n\n".join(set(retrieved_documents)) if retrieved_documents else "해당 카테고리에 대한 추가 정보는 제공되지 않습니다."
        context = f"대화 기록:\n{history}\n질문: {query}\n카테고리: {category}\n의도: {intent}\n\n{retrieved_text}"
        return context

    def update_conversation_history(self, query, answer, retrieved_documents):
        """
        대화 이력을 업데이트하고, 최대 토큰 수에 맞게 조절합니다.

        Parameters:
            query (str): 사용자 질문
            answer (str): 생성된 답변
            retrieved_documents (list): 검색된 문서 리스트
        """
        self.conversation_history.append(f"질문: {query}")
        self.conversation_history.append(f"답변: {answer}")
        if retrieved_documents:
            documents_text = "\n".join(set(retrieved_documents))
            self.conversation_history.append(f"조회된 문서:\n{documents_text}")

        history_text = "\n".join(self.conversation_history)
        max_history_tokens = 2048
        truncated_history = truncate_history(history_text, max_tokens=max_history_tokens)

        self.conversation_history = truncated_history.split("\n")

    def identify_category(self, query, faqs_context=None):
        """
        질문에 대한 카테고리를 식별합니다.

        Parameters:
            query (str): 사용자 질문
            faqs_context (str, optional): FAQ 컨텍스트

        Returns:
            str: 식별된 카테고리
        """
        system_prompt = self.category_prompt.format(context=faqs_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        response = self.language_model.generate(messages).strip()
        return response

    def understand_intent(self, query, category, faqs_context=None):
        """
        질문의 의도를 파악합니다.

        Parameters:
            query (str): 사용자 질문
            category (str): 식별된 카테고리
            faqs_context (str, optional): FAQ 컨텍스트

        Returns:
            str: 파악된 의도
        """
        system_prompt = self.intent_prompt.format(context=faqs_context, category=category)
        prompt = f"질문: '{query}'\n카테고리: '{category}'"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.language_model.generate(messages).strip()
        return response

    def retrieve_documents(self, query, n_results):
        """
        문서를 검색합니다.

        Parameters:
            query (str): 사용자 질문
            n_results (int): 검색할 문서 수

        Returns:
            list: 검색된 문서 리스트
        """
        results = self.retriever.retrieve(query, n_results)
        return results

    def generate_answer(self, query, category, intent, retrieved_documents):
        """
        최종 답변을 생성합니다.

        Parameters:
            query (str): 사용자 질문
            category (str): 식별된 카테고리
            intent (str): 식별된 의도
            retrieved_documents (list): 검색된 문서 리스트

        Returns:
            str: 생성된 답변
        """
        if intent == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.':
            return intent

        history = "\n".join(set(self.conversation_history))
        retrieved_text = "\n\n".join(set(retrieved_documents)) if retrieved_documents else "해당 카테고리에 대한 추가 정보는 제공되지 않습니다."

        system_prompt = self.answer_prompt.format(
            context=retrieved_text,
            category=category,
            intent=intent,
            history=history
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        answer = self.language_model.generate(messages).strip()

        self.update_conversation_history(query, answer, retrieved_documents)
        return f"카테고리: {category}\n의도: {intent}\n\n{answer}"
