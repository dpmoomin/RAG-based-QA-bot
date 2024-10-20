# main.py
import pickle
from utils.preprocess import preprocess_qa_data

def extract_questions_and_answers(file_path):
    try:
        # 데이터 로드
        with open(file_path, 'rb') as f:
            faq_data = pickle.load(f)

        # 전처리된 질문과 답변을 얻음
        qa_pairs = preprocess_qa_data(faq_data)

        return qa_pairs

    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return []
    except Exception as e:
        print(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return []
