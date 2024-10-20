from tqdm import tqdm

class FAQTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, separator_pattern='\n\n'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_pattern = separator_pattern

    def split(self, faq_data):
        """
        FAQ 데이터를 문서로 변환하고 분할합니다. 질문과 답변을 하나의 텍스트로 합쳐 저장합니다.

        Parameters:
            faq_data (list): FAQ 데이터 리스트. 각 항목은 {'question': ..., 'answer': ...} 형식의 딕셔너리입니다.

        Returns:
            list: 분할된 문서 청크 리스트 및 각 청크의 메타데이터.
        """
        documents = []
        metadatas = []

        # FAQ 데이터에서 각 질문과 답변을 합쳐서 분할
        for item in tqdm(faq_data, desc="Processing FAQ Data..."):
            question = item.get('question', '')
            answer = item.get('answer', '')

            # 질문과 답변을 하나의 텍스트로 합침
            combined_text = f"Q: {question}\nA: {answer}"
            
            # 분할된 청크와 메타데이터 생성
            chunks = self.split_document(combined_text)
            documents.extend(chunks)
            metadatas.extend([{'question': question}] * len(chunks))

        return documents, metadatas

    def split_document(self, document):
        """
        단일 문서를 청크로 분할합니다.

        Parameters:
            document (str): 단일 문서 텍스트.

        Returns:
            list: 청크 리스트.
        """
        if not document:
            return []

        splits = document.split(self.separator_pattern)
        chunks = []
        current_chunk = splits[0]

        for split in tqdm(splits[1:], desc="Splitting Document..."):
            if len(current_chunk) + len(split) + len(self.separator_pattern) > self.chunk_size:
                chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + self.separator_pattern + split
            else:
                current_chunk += self.separator_pattern + split

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
