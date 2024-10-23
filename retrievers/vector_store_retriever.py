class VectorStoreRetriever:
    def __init__(self, vector_store, k=4, threshold=0.35):
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

    def retrieve(self, query, n_results):
        """
        주어진 질의에 대한 유사한 문서를 검색합니다.

        Parameters:
            query (str): 검색할 질의.

        Returns:
            list: 검색된 문서 리스트 또는 None.
        """
        results = self.vector_store.similarity_search(query, n_results, threshold=self.threshold)
        
        # 필터링된 결과 반환
        return [result['text'] for result in results] if results else None
