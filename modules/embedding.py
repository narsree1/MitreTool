import torch
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    """
    Load the sentence transformer model for embeddings
    
    Returns:
        model: The loaded SentenceTransformer model
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Using BAAI/bge-large-en-v1.5, one of the top embedding models on the MTEB leaderboard
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        model = model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        # Fallback to a smaller but still high-quality model
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            model = model.to(device)
            return model
        except Exception as fallback_error:
            st.error(f"Error loading fallback model: {fallback_error}")
            return None

@st.cache_resource
def get_mitre_embeddings(_model, techniques):
    """
    Generate embeddings for MITRE technique descriptions
    
    Args:
        _model: SentenceTransformer model
        techniques: List of MITRE technique dictionaries
        
    Returns:
        embeddings: Tensor of embeddings for technique descriptions
    """
    if _model is None or not techniques:
        return None
    try:
        descriptions = [tech['description'] for tech in techniques]
        
        # Encode all descriptions in batches - smaller batch size for larger models
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i+batch_size]
            # BGE models work best with normalize_embeddings=True
            batch_embeddings = _model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {e}")
        return None

def cosine_similarity_search(query_embedding, reference_embeddings):
    """
    Perform cosine similarity search between a query embedding and reference embeddings
    
    Args:
        query_embedding: Embedding vector for the query
        reference_embeddings: Tensor of reference embedding vectors
        
    Returns:
        best_score: Highest similarity score
        best_idx: Index of the best matching reference embedding
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)
    if not isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = torch.tensor(reference_embeddings)
    
    # Ensure query_embedding is 1D if it's just one embedding
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.unsqueeze(0)
    
    # Normalize the embeddings - BGE models already normalize during encoding,
    # but we'll normalize again just to be safe
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity
    similarities = torch.mm(query_embedding, reference_embeddings.T)
    
    # Get the best match
    best_idx = similarities[0].argmax().item()
    best_score = similarities[0][best_idx].item()
    
    return best_score, best_idx

def batch_similarity_search(query_embeddings, reference_embeddings):
    """
    Perform batch cosine similarity search
    
    Args:
        query_embeddings: Tensor of query embedding vectors
        reference_embeddings: Tensor of reference embedding vectors
        
    Returns:
        best_scores: List of highest similarity scores for each query
        best_indices: List of indices of the best matching reference embedding for each query
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings)
    if not isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = torch.tensor(reference_embeddings)
    
    # Normalize the embeddings
    query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity
    similarities = torch.mm(query_embeddings, reference_embeddings.T)
    
    # Get the best matches for each query
    best_scores, best_indices = similarities.max(dim=1)
    
    return best_scores.tolist(), best_indices.tolist()
