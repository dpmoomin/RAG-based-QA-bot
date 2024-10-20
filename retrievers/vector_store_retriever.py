class VectorStoreRetriever:
    def __init__(self, vector_store, k=4, threshold=0.5):
        """
        초기화 메서드입니다.

        Parameters:
            vector_store: 벡터 저장소.
            k (int): 검색할 문서의 개수.
            threshold (float): 유사도 점수의 임계값.
        """
        self.vector_store = vector_store
        self.k = k
        self.threshold = threshold

    def retrieve(self, query):
        """
        주어진 질의에 대한 유사한 문서를 검색합니다.

        Parameters:
            query (str): 검색할 질의.

        Returns:
            list: 검색된 문서 리스트.
        """
        results = self.vector_store.similarity_search(query, n_results=self.k)
        
        # 유사도 점수 확인
        if results and results[0].get('score', 0) < self.threshold:
            return None  # 유사도 점수가 임계값보다 낮을 때 None 반환
        
        return results