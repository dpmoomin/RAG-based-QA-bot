from prompts.prompt_templates import (
    DEFAULT_SYSTEM_PROMPT,
    CATEGORY_IDENTIFICATION_PROMPT,
    INTENT_UNDERSTANDING_PROMPT
)
from models.language_model import OpenAILanguageModel
from config.settings import OPENAI_API_KEY
import tiktoken

# 토큰 계산 함수
def count_tokens(text, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# 대화 이력 길이 제한 함수
def truncate_history(history, max_tokens=2048, encoding_name='cl100k_base'):
    # 전체 토큰 수 계산
    current_tokens = count_tokens(history, encoding_name)
    
    # 최대 토큰 수를 초과하는 경우
    if current_tokens > max_tokens:
        # 대화 이력을 '\n'을 기준으로 분할
        messages = history.split("\n")
        truncated_history = ""
        
        # 토큰 수를 줄여가며 최신 메시지부터 추가
        for message in reversed(messages):
            truncated_history = message + "\n" + truncated_history
            current_tokens = count_tokens(truncated_history, encoding_name)
            if current_tokens > max_tokens:
                break
        # 초과된 메시지들을 제거한 결과 반환
        return truncated_history.strip()
    
    # 최대 토큰 수를 초과하지 않으면 그대로 반환
    return history

class RetrievalQAChain:
    def __init__(self, retriever, language_model=None):
        self.retriever = retriever
        self.category_prompt = CATEGORY_IDENTIFICATION_PROMPT  # 카테고리 식별 프롬프트
        self.intent_prompt = INTENT_UNDERSTANDING_PROMPT       # 의도 파악 프롬프트
        self.answer_prompt = DEFAULT_SYSTEM_PROMPT             # 답변 생성 프롬프트
        self.language_model = language_model or OpenAILanguageModel(api_key=OPENAI_API_KEY)
        self.conversation_history = []  # 대화 기록을 저장하기 위한 리스트

    def run(self, query):
        # Step 1: 문서 검색
        retrieved_documents = self.retrieve_documents(query, 5)

        # Step 2: 카테고리 식별
        category = self.identify_category(query, retrieved_documents)
        if category == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.':
            return category
        elif ('•' in category or '-' in category) and (category.count('•') > 1 or category.count('-') > 1):
            # 카테고리가 불명확할 경우, 예상 카테고리를 제시하고 사용자에게 선택 요청
            print("\n카테고리가 불명확합니다. 아래의 옵션 중에서 선택해 주세요:\n")
            print(category)  # Display the generated bullet points
            category = input("답변: ").strip()  # Use the user's response directly

        # Step 3: 질문의 의도 파악
        intent = self.understand_intent(query, category, retrieved_documents)
        if intent == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.':
            return intent
        elif ('•' in intent or '-' in intent) and (intent.count('•') > 1 or intent.count('-') > 1):
            # 의도가 불명확할 경우, 예상 의도를 제시하고 사용자에게 선택 요청
            print("\n의도가 불명확합니다. 아래의 옵션 중에서 선택해 주세요:\n")
            print(intent)  # Display the generated bullet points
            intent = input("답변: ").strip()  # Use the user's response directly
        
        # Step 4: 답변 생성
        answer = self.generate_answer(query, category, intent, retrieved_documents)

        # Update conversation history
        self.update_conversation_history(query, answer, retrieved_documents)
        return answer

    def build_context(self, query, category, intent, retrieved_documents):
        # 중복된 내용 제거하여 대화 기록과 조회된 문서로 구성된 컨텍스트 생성
        history = "\n".join(set(self.conversation_history))
        retrieved_text = "\n\n".join(set(retrieved_documents)) if retrieved_documents else "해당 카테고리에 대한 추가 정보는 제공되지 않습니다."
        context = f"대화 기록:\n{history}\n질문: {query}\n카테고리: {category}\n의도: {intent}\n\n{retrieved_text}"
        return context

    # 기존의 update_conversation_history 메서드에 history 조절 로직 추가
    def update_conversation_history(self, query, answer, retrieved_documents):
        # Update conversation history with the latest query, answer, and retrieved documents
        self.conversation_history.append(f"질문: {query}")
        self.conversation_history.append(f"답변: {answer}")
        if retrieved_documents:
            documents_text = "\n".join(set(retrieved_documents))  # 중복된 문서 제거
            self.conversation_history.append(f"조회된 문서:\n{documents_text}")

        # 대화 이력 길이를 조절
        history_text = "\n".join(self.conversation_history)
        max_history_tokens = 2048  # history에 할당할 최대 토큰 수
        truncated_history = truncate_history(history_text, max_tokens=max_history_tokens)

        # 조절된 대화 이력을 리스트 형태로 다시 저장
        self.conversation_history = truncated_history.split("\n")

    def identify_category(self, query, faqs_context=None):
        # 카테고리 식별을 위한 프롬프트 생성
        system_prompt = self.category_prompt.format(context=faqs_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        response = self.language_model.generate(messages).strip()
        return response

    def understand_intent(self, query, category, faqs_context=None):
        # 의도 파악을 위한 프롬프트 생성
        system_prompt = self.intent_prompt.format(context=faqs_context, category=category)
        prompt = f"질문: '{query}'\n카테고리: '{category}'"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.language_model.generate(messages).strip()
        return response

    def retrieve_documents(self, query, n_results):
        # Query with category and intent
        combined_query = f"{query}"
        results = self.retriever.retrieve(combined_query, n_results)
        return results

    def generate_answer(self, query, category, intent, retrieved_documents):
        if intent == '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.':
            return intent
        # 중복된 대화 기록과 문서를 제거하여 구성
        history = "\n".join(set(self.conversation_history))  # 중복된 대화 기록 제거
        retrieved_text = "\n\n".join(set(retrieved_documents)) if retrieved_documents else "해당 카테고리에 대한 추가 정보는 제공되지 않습니다."

        # 답변 생성을 위한 프롬프트 생성
        system_prompt = self.answer_prompt.format(context=retrieved_text, category=category, intent=intent, history=history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        # 답변 생성
        answer = self.language_model.generate(messages).strip()

        # 대화 기록 업데이트
        self.update_conversation_history(query, answer, retrieved_documents)
        return f"카테고리: {category}\n의도: {intent}\n\n{answer}"
