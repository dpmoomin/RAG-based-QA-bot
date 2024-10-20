# embeddings.py
from openai import OpenAI

class OpenAIEmbedding:
    def __init__(self, api_key, model="text-embedding-3-small"):
        """
        Initializes the OpenAIEmbedding model.
        
        Parameters:
            api_key (str): OpenAI API key.
            model (str): The embedding model to use.
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def get_embedding(self, text):
        """
        Generates an embedding for a single text input using the OpenAI API.
        
        Parameters:
            text (str): The input text to be embedded.
        
        Returns:
            list: The embedding vector for the input text.
        """
        try:
            response = self.client.embeddings.create(
                input=[text],  # Send as a list
                model=self.model  # Use the specified embedding model
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"[Error] Failed to generate embedding: {e}")
            return []
