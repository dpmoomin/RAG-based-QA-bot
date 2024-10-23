import pickle

class FAQDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        return data
