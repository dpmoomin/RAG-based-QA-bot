import re
import logging
from konlpy.tag import Okt
from constants import STOPWORDS, QUESTION_RELATED_STOPWORDS

logger = logging.getLogger(__name__)

# KoNLPy의 Okt 토크나이저 초기화
okt = Okt()

def normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다.

    Args:
        text (str): 정규화할 텍스트.

    Returns:
        str: 정규화된 텍스트.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(tokens: list, stopwords: set) -> list:
    """
    토큰 리스트에서 불용어를 제거합니다.

    Args:
        tokens (list): 토큰의 리스트.
        stopwords (set): 불용어 집합.

    Returns:
        list: 불용어가 제거된 토큰의 리스트.
    """
    return [word for word in tokens if word not in stopwords]

def preprocess_text(text: str) -> str:
    """
    입력된 텍스트를 전처리합니다.

    Args:
        text (str): 전처리할 텍스트.

    Returns:
        str: 전처리된 텍스트.
    """
    # 1. 텍스트 정규화
    normalized_text = normalize_text(text)

    # 2. 한국어 전용 토크나이저 (Okt) 사용
    tokens = okt.morphs(normalized_text)

    # 3. 불용어 제거
    combined_stopwords = set(STOPWORDS + QUESTION_RELATED_STOPWORDS)
    filtered_tokens = remove_stopwords(tokens, combined_stopwords)

    # 4. 토큰 재조합
    processed_text = ' '.join(filtered_tokens)

    return processed_text

def preprocess_qa_data(faq_data: dict) -> list:
    """
    질문과 답변 데이터를 전처리합니다.

    Args:
        faq_data (dict): FAQ 데이터. 각 항목은 {'질문': ..., '답변': ...} 형식입니다.

    Returns:
        list: 전처리된 질문과 답변의 리스트.
    """
    # 질문과 답변을 저장할 리스트
    qa_pairs = []

    # 답변 끝의 불필요한 부분을 제거하기 위한 패턴
    unwanted_text_pattern = re.compile(r'위 도움말이 도움이 되었나요\?.*', re.DOTALL)

    for question, answer in faq_data.items():
        # 질문과 답변에서 불필요한 특수문자 제거 및 전처리
        cleaned_question = re.sub(r'\\[a-zA-Z0-9]+', '', question)
        cleaned_question = preprocess_text(cleaned_question)

        cleaned_answer = re.sub(unwanted_text_pattern, '', answer).strip()
        cleaned_answer = re.sub(r'\\[a-zA-Z0-9]+', '', cleaned_answer)
        cleaned_answer = preprocess_text(cleaned_answer)

        # 정제된 질문과 답변을 리스트에 추가
        qa_pairs.append({'question': cleaned_question, 'answer': cleaned_answer})

    return qa_pairs
