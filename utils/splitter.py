from tqdm import tqdm

class FAQTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, separator_pattern='\n\n'):
        """
        FAQ 텍스트를 분할하는 클래스 초기화.

        Parameters:
            chunk_size (int): 청크의 최대 길이
            chunk_overlap (int): 청크 간의 중첩 길이
            separator_pattern (str): 분할을 위한 구분자 패턴
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_pattern = separator_pattern

    def split(self, faq_data):
        """
        FAQ 데이터를 청크로 분할합니다.

        Parameters:
            faq_data (list): FAQ 데이터 리스트

        Returns:
            tuple: 분할된 문서 리스트와 메타데이터 리스트
        """
        documents = []
        metadatas = []

        for item in tqdm(faq_data, desc="FAQ 데이터 처리 중"):
            question = item.get('question', '')
            answer = item.get('answer', '')

            combined_text = f"Q: {question}\nA: {answer}"

            chunks = self.split_document(combined_text)
            documents.extend(chunks)
            metadatas.extend([{'question': question}] * len(chunks))

        return documents, metadatas

    def split_document(self, document):
        """
        단일 문서를 청크로 분할합니다.

        Parameters:
            document (str): 문서 텍스트

        Returns:
            list: 청크 리스트
        """
        if not document:
            return []

        splits = document.split(self.separator_pattern)
        chunks = []
        current_chunk = splits[0]

        for split in tqdm(splits[1:], desc="문서 분할 중"):
            if len(current_chunk) + len(split) + len(self.separator_pattern) > self.chunk_size:
                chunks.append(current_chunk.strip())
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + self.separator_pattern + split
            else:
                current_chunk += self.separator_pattern + split

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
