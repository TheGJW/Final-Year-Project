from sentence_transformers import SentenceTransformer

class SemanticEmbeddingModel:
    # initialize and loading the model 
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    # function to convert input texts into embedding vectors
    def encode(self, texts):
        return self.model.encode(texts)

