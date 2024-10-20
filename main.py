from config.settings import OPENAI_API_KEY
from stores.chroma_vector_store import ChromaVectorStore
from retrievers.vector_store_retriever import VectorStoreRetriever
from chains.retrieval_qa_chain import RetrievalQAChain

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일에 'OPENAI_API_KEY'를 설정하세요.")

    # 벡터 스토어 구성
    vector_store = ChromaVectorStore(api_key=OPENAI_API_KEY)

    # Chroma DB에서 저장된 문서 불러오기
    saved_documents = vector_store.load_documents()
    if not saved_documents:
        raise ValueError("Chroma DB에 저장된 임베딩 데이터가 없습니다. 먼저 embed_and_store.py를 실행하세요.")

    # 검색기 구성
    retriever = VectorStoreRetriever(vector_store, k=3, threshold=0.5)

    # 질의응답 체인 구성
    qa_chain = RetrievalQAChain(retriever)

    # 사용자 질의 응답 루프
    print("안녕하세요! 네이버 스마트스토어에 관한 질문을 도와드리는 챗봇입니다. 궁금하신 내용을 물어보세요.")
    while True:
        query = input("질문: ")
        if query.lower() == 'exit':
            break
        
        # 검색 결과 확인
        retrieved_documents = retriever.retrieve(query)
        if not retrieved_documents:
            print("\n저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.\n")
        else:
            # 검색된 결과를 사용하여 답변 생성
            answer = qa_chain.run(query)
            print("\n답변:")
            print(answer)
            print("\n")

if __name__ == "__main__":
    main()