import chromadb
from chromadb.utils import embedding_functions
from embeddings.embedding import OpenAIEmbedding
import traceback
import logging
import os
import json
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaVectorStore:
    def __init__(self, api_key, persist_directory="chroma_db", embedding_model="text-embedding-3-small", batch_size=1, progress_file="progress.json"):
        """
        ChromaVectorStore 초기화
        
        Parameters:
            api_key (str): OpenAI API 키.
            persist_directory (str): 로컬에 데이터 저장 경로.
            embedding_model (str): 사용할 임베딩 모델.
            batch_size (int): 한 번에 처리할 최대 문서 수 (기본값: 100).
            progress_file (str): 진행 상태를 저장할 파일 이름.
        """
        self.api_key = api_key
        self.embedding_model = OpenAIEmbedding(api_key, model=embedding_model)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_collection(name="faq_collection", embedding_function=self.embedding_function)
        self.batch_size = batch_size
        self.progress_file = progress_file
        self.load_progress()
        logging.info("ChromaVectorStore가 초기화되었습니다. 임베딩 모델: %s", embedding_model)

    def load_progress(self):
        """
        이전의 진행 상태를 불러옵니다.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as file:
                self.progress = json.load(file)
            logging.info("진행 상태가 로드되었습니다. 마지막 인덱스: %d", self.progress.get("last_index", 0))
        else:
            self.progress = {"last_index": 0}
            logging.info("새로운 진행 상태가 시작됩니다.")

    def save_progress(self, last_index):
        """
        진행 상태를 저장합니다.
        
        Parameters:
            last_index (int): 마지막으로 성공적으로 저장된 인덱스.
        """
        self.progress["last_index"] = last_index
        with open(self.progress_file, "w") as file:
            json.dump(self.progress, file)
        logging.info("진행 상태가 저장되었습니다. 마지막 인덱스: %d", last_index)

    def add_documents(self, documents, metadatas=None, ids=None):
        """
        문서를 Chroma DB에 추가합니다. 이미 저장된 문서는 건너뜁니다.
        중간에 실패해도 이전까지 저장된 내용은 유지되고, 다음 실행 시 실패한 부분부터 다시 시도합니다.

        Parameters:
            documents (list): 문서 리스트.
            metadatas (list, optional): 각 문서의 메타데이터 리스트.
            ids (list, optional): 각 문서의 고유 ID 리스트.
        """
        if not documents or not all(isinstance(doc, str) for doc in documents):
            logging.error("유효한 문서 리스트를 제공해야 합니다.")
            return

        try:
            # 시작 인덱스를 진행 상태에서 불러옴
            start_index = self.progress.get("last_index", 0)
            documents = documents[start_index:]
            ids = ids[start_index:] if ids else None
            metadatas = metadatas[start_index:] if metadatas else None

            # 이미 존재하는 문서 ID 확인
            existing_ids = set(metadata['id'] for metadata in self.collection.get().get('metadatas', []) if 'id' in metadata)

            new_documents, new_ids, new_metadatas = [], [], []
            for idx, doc in enumerate(documents):
                doc_id = ids[idx] if ids else str(start_index + idx)
                if doc_id not in existing_ids:
                    new_documents.append(doc)
                    new_ids.append(doc_id)
                    if metadatas:
                        new_metadatas.append(metadatas[idx])

                # 배치 사이즈만큼 문서가 준비되면 추가
                if len(new_documents) >= self.batch_size:
                    self._try_add_documents(new_documents, new_ids, new_metadatas, start_index + idx)
                    new_documents, new_ids, new_metadatas = [], [], []

            # 남은 문서 추가
            if new_documents:
                self._try_add_documents(new_documents, new_ids, new_metadatas, start_index + len(new_documents) - 1)
            else:
                logging.info("추가할 새 문서가 없습니다.")
        except Exception as e:
            logging.error("문서 추가 중 오류가 발생했습니다: %s", e)
            logging.debug("예외 정보: %s", traceback.format_exc())

    def _try_add_documents(self, documents, ids, metadatas, last_index):
        """
        문서 리스트를 추가합니다. 예외 발생 시에도 다음 배치로 진행합니다.

        Parameters:
            documents (list): 문서 리스트.
            ids (list): 문서의 ID 리스트.
            metadatas (list, optional): 메타데이터 리스트.
            last_index (int): 마지막으로 성공적으로 저장된 인덱스.
        """
        try:
            self._add_documents(documents, ids, metadatas)
            # 진행 상태 저장
            self.save_progress(last_index)
        except Exception as e:
            logging.error("배치 추가 중 오류가 발생했습니다. 다음 배치로 계속 진행합니다: %s", e)
            logging.debug("예외 정보: %s", traceback.format_exc())

    def _add_documents(self, documents, ids, metadatas):
        """
        문서 리스트를 추가합니다.

        Parameters:
            documents (list): 문서 리스트.
            ids (list): 문서의 ID 리스트.
            metadatas (list, optional): 메타데이터 리스트.
        """
        valid_documents = []
        valid_ids = []
        valid_metadatas = []
        embeddings = []

        # 모든 문서의 임베딩을 생성하고 유효한 문서만 추가
        for idx, doc in enumerate(tqdm(documents, desc="Generating Embeddings")):
            embedding = self.embedding_model.get_embedding(doc)

            # 임베딩이 비어 있거나 형식이 올바르지 않으면 건너뜀
            if not embedding or not isinstance(embedding, list):
                logging.warning("문서 %d의 임베딩이 비어 있거나 형식이 잘못되었습니다. 텍스트: %s", idx, doc)
                continue

            # 모든 임베딩 크기가 동일한지 확인
            if embeddings and len(embedding) != len(embeddings[0]):
                logging.warning("문서 %d의 임베딩 크기가 일관되지 않습니다. 텍스트: %s", idx, doc)
                continue

            embeddings.append(embedding)
            valid_documents.append(doc)
            valid_ids.append(ids[idx])
            if metadatas:
                valid_metadatas.append(metadatas[idx])

        if not embeddings:
            logging.error("유효한 임베딩이 없습니다. 문서 추가를 중단합니다.")
            return

        # 유효한 문서들만 추가
        try:
            self.collection.add(
                documents=valid_documents,
                ids=valid_ids,
                metadatas=valid_metadatas,
                embeddings=embeddings
            )
            logging.info("%d개의 문서가 성공적으로 추가되었습니다.", len(valid_documents))
        except Exception as e:
            logging.error("Chroma DB에 문서 추가 중 오류가 발생했습니다. 일부 문서를 건너뜁니다: %s", e)
            logging.debug("예외 정보: %s", traceback.format_exc())

    def load_documents(self):
        """
        Chroma DB에서 저장된 문서를 불러옵니다.

        Returns:
            list: 저장된 문서 리스트.
        """
        try:
            self.collection = self.client.get_collection(name="faq_collection", embedding_function=self.embedding_function)
            results = self.collection.get(include=["documents", "metadatas"])
            documents = results.get('documents', [])
            if documents:
                logging.info("%d개의 문서가 불러와졌습니다.", len(documents))
            else:
                logging.info("저장된 문서가 없습니다.")
            return documents
        except Exception as e:
            logging.error("문서 불러오기 중 오류가 발생했습니다: %s", e)
            logging.debug("예외 정보: %s", traceback.format_exc())
            return []

    def similarity_search(self, query, n_results=3, threshold=0.42):
        """
        질의에 대한 유사한 문서를 검색합니다.

        Parameters:
            query (str): 검색할 질의.
            n_results (int): 반환할 결과 수.
            threshold (float): 유사도 점수의 임계값.

        Returns:
            list: 유사도 점수가 임계값을 넘는 문서 리스트.
        """
        try:
            # 데이터베이스에서 결과 가져오기
            results = self.collection.query(query_texts=[query], n_results=n_results)
            logging.debug("Raw query results: %s", results)

            filtered_results = []
            if results and 'documents' in results and results['documents'] and 'distances' in results and results['distances']:
                for i, doc in enumerate(results['documents'][0]):
                    # 거리 값을 가져와 유사도로 변환 (1 / (1 + distance))
                    distance = results['distances'][0][i]
                    similarity_score = 1 / (1 + distance)

                    # 모든 문서의 유사도를 로그로 기록
                    logging.info("문서: %s, 유사도: %.4f", doc[:100], similarity_score)

                    # 임계값을 넘는 유사도만 필터링
                    if similarity_score >= threshold:
                        doc_with_score = {'text': doc, 'score': similarity_score}
                        filtered_results.append(doc_with_score)

                logging.info("임계값 %.2f 이상인 유사한 문서 %d개를 찾았습니다.", threshold, len(filtered_results))
            else:
                logging.info("문서를 찾지 못했습니다.")
            return filtered_results
        except Exception as e:
            logging.error("유사도 검색 중 오류가 발생했습니다: %s", e)
            logging.debug("예외 정보: %s", traceback.format_exc())
            return []
