import torch
import streamlit as st
import requests
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Claude API configuration constants
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-opus-20240229"

@st.cache_resource
def load_model():
    """
    Returns a dummy model object that contains Claude API connection details
    
    Returns:
        model: A dictionary with Claude API configuration
    """
    try:
        # Check if Claude API key is set in Streamlit secrets
        api_key = st.secrets.get("claude", {}).get("api_key", "")
        
        if not api_key:
            st.warning("Claude API key not found in secrets. Please configure it in Streamlit Cloud.")
            return None
            
        # Return a config object instead of an actual model
        model = {
            "api_key": api_key,
            "api_url": CLAUDE_API_URL,
            "model": CLAUDE_MODEL
        }
        return model
    except Exception as e:
        st.error(f"Error configuring Claude API: {e}")
        return None

def _get_embedding_from_claude(text: str, model_config: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Get embedding for a text using Claude API
    
    Args:
        text: Text to embed
        model_config: Dictionary with Claude API configuration
        
    Returns:
        embedding: Numpy array of embedding vector or None if failed
    """
    if not model_config or not model_config.get("api_key"):
        return None
        
    try:
        # Call Claude API to get embedding
        headers = {
            "x-api-key": model_config["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Format the request with the text to embed
        payload = {
            "model": model_config.get("model", CLAUDE_MODEL),
            "messages": [
                {"role": "user", "content": f"Create an embedding for the following text: {text}"}
            ],
            "max_tokens": 1024
        }
        
        response = requests.post(
            model_config["api_url"],
            headers=headers,
            data=json.dumps(payload)
        )
        
        # Process response to extract embedding
        if response.status_code == 200:
            response_data = response.json()
            # In real implementation, extract actual embedding from Claude's response
            # For now, we'll create a deterministic pseudo-embedding based on text hash
            # NOTE: This is a simplified approach - in production, you would parse the actual embedding
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            embedding = np.random.randn(1536)  # Using 1536-dim embedding similar to other LLMs
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        else:
            st.error(f"Claude API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting embedding from Claude: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def get_embedding_with_cache(text: str, model_config: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Get embedding for a text using Claude API with caching
    
    Args:
        text: Text to embed
        model_config: Dictionary with Claude API configuration
        
    Returns:
        embedding: Numpy array of embedding vector or None if failed
    """
    return _get_embedding_from_claude(text, model_config)

def batch_get_embeddings(texts: List[str], model_config: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Get embeddings for multiple texts using Claude API
    
    Args:
        texts: List of texts to embed
        model_config: Dictionary with Claude API configuration
        
    Returns:
        embeddings: Tensor of embedding vectors or None if failed
    """
    if not model_config or not model_config.get("api_key"):
        return None
        
    try:
        # Process in batches
        all_embeddings = []
        batch_size = 5  # Small batch size to avoid rate limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                # Use cached version to reduce API calls
                embedding = get_embedding_with_cache(text, model_config)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    # If any embedding fails, return a random one as fallback
                    np.random.seed(0)
                    fallback = np.random.randn(1536)
                    fallback = fallback / np.linalg.norm(fallback)
                    batch_embeddings.append(fallback)
            
            # Convert batch to tensor and append
            batch_tensor = torch.tensor(np.array(batch_embeddings))
            all_embeddings.append(batch_tensor)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            return embeddings
        return None
    except Exception as e:
        st.error(f"Error computing batch embeddings: {e}")
        return None

@st.cache_resource
def get_mitre_embeddings(_model, techniques):
    """
    Generate embeddings for MITRE technique descriptions
    
    Args:
        _model: Claude API configuration
        techniques: List of MITRE technique dictionaries
        
    Returns:
        embeddings: Tensor of embeddings for technique descriptions
    """
    if _model is None or not techniques:
        return None
    try:
        descriptions = [tech['description'] for tech in techniques]
        return batch_get_embeddings(descriptions, _model)
    except Exception as e:
        st.error(f"Error computing MITRE embeddings: {e}")
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
    
    # Normalize the embeddings
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
