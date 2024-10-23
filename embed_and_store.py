from config.settings import OPENAI_API_KEY
from utils.extracter import extract_questions_and_answers
from utils.splitter import FAQTextSplitter
from stores.chroma_vector_store import ChromaVectorStore

def embed_and_store(file_path):
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 'OPENAI_API_KEY'를 설정하세요.")

    # 질문과 답변 추출
    qa_pairs = extract_questions_and_answers(file_path)

    if not qa_pairs:
        print("질문과 답변 데이터가 없습니다.")
        return

    # 텍스트 스플리터 초기화
    text_splitter = FAQTextSplitter(chunk_size=256, chunk_overlap=0)
    documents, metadatas = text_splitter.split(qa_pairs)

    # 벡터 스토어 구성
    vector_store = ChromaVectorStore(api_key=OPENAI_API_KEY)

    # 문서를 벡터 스토어에 추가 (임베딩 및 저장)
    ids = [str(i) for i in range(len(documents))]
    vector_store.add_documents(documents, metadatas, ids)

    print("데이터 임베딩 및 저장이 완료되었습니다. Chroma DB에 문서가 저장되었습니다.")

if __name__ == "__main__":
    embed_and_store('datasets/final_result.pkl')