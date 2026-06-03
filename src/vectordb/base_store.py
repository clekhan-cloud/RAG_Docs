from abc import ABC, abstractmethod


class BaseVectorStore(ABC):

    @abstractmethod
    def add_documents(self, texts, metadatas):
        pass

    @abstractmethod
    def search(self, query, k=3):
        pass

    @abstractmethod
    def save_index(self):
        pass

    @abstractmethod
    def load_index(self):
        pass