from sentence_transformers import SentenceTransformer

# Load embedding model (this is your ML model)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text: str):
    """
    Convert text into a vector embedding.
    This is the core ML step of your project.
    """
    if not text or not text.strip():
        raise ValueError("Text for embedding cannot be empty")

    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()
