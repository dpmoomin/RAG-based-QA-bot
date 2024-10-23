from config.settings import OPENAI_API_KEY
from stores.chroma_vector_store import ChromaVectorStore
from retrievers.vector_store_retriever import VectorStoreRetriever
from chains.retrieval_qa_chain import RetrievalQAChain

def main():
    """
    메인 함수: 사용자 질의에 대한 응답을 제공합니다.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 'OPENAI_API_KEY'를 설정하세요.")

    vector_store = ChromaVectorStore(api_key=OPENAI_API_KEY)

    saved_documents = vector_store.load_documents()
    if not saved_documents:
        raise ValueError("Chroma DB에 저장된 임베딩 데이터가 없습니다. 먼저 embed_and_store.py를 실행하세요.")

    retriever = VectorStoreRetriever(vector_store, k=3, threshold=0.35)

    qa_chain = RetrievalQAChain(retriever)

    print("안녕하세요.\n\n궁금한 내용을 간단히 입력해 주시면 도움을 드릴게요!\n\n예) 스마트스토어센터 가입 절차, 상품등록 방법, 발송 처리 기한 등")
    while True:
        query = input("질문: ")
        if query.lower() == 'exit':
            break

        answer = qa_chain.run(query)
        print("\n답변:")
        print(answer)
        print("\n")

if __name__ == "__main__":
    main()
