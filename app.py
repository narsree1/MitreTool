# app.py - Main MITRE ATT&CK Mapping Tool

import pandas as pd
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json
import datetime
import uuid
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import time
import os
from typing import List, Dict, Tuple, Any, Optional
import anthropic  # Added for Claude Haiku API

# Import the modules
from analytics import render_analytics_page
from suggestions import render_suggestions_page, get_suggested_use_cases

# App configuration
st.set_page_config(
    page_title="MITRE ATT&CK Mapping Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define fallback similarity search functions
def cosine_similarity_search(query_embedding, reference_embeddings):
    """
    Fallback similarity search using PyTorch tensors
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
    Batch similarity search using PyTorch tensors
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

# Custom CSS for modern look
st.markdown("""
<style>
    /* Modern Color Scheme */
    :root {
        --primary: #0d6efd;
        --secondary: #6c757d;
        --success: #198754;
        --danger: #dc3545;
        --warning: #ffc107;
        --info: #0dcaf0;
        --background: #f8f9fa;
        --card-bg: #ffffff;
        --text: #212529;
    }
    
    /* Main elements */
    .main {
        background-color: var(--background);
        padding: 1.5rem;
    }
    
    /* Cards styling */
    .card {
        background-color: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Modern button styles */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Header styling - reduced font size */
    h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        font-size: 1.8rem;
    }
    
    h2 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
    }
    
    h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Smaller text everywhere */
    .stMarkdown, p, div, span, .stText {
        font-size: 0.9rem;
    }
    
    /* Upload area styling */
    .uploadfile {
        border: 2px dashed #0d6efd;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        background-color: rgba(13, 110, 253, 0.05);
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
        padding: 12px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 12px;
        color: var(--secondary);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.85rem;
    }
    
    /* Improve default slider styling */
    .stSlider div[data-baseweb="slider"] {
        height: 5px;
    }
    
    /* Sidebar text smaller */
    .sidebar .stMarkdown {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'techniques_count' not in st.session_state:
    st.session_state.techniques_count = {}
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'mapping_complete' not in st.session_state:
    st.session_state.mapping_complete = False
if 'library_data' not in st.session_state:
    st.session_state.library_data = None
if 'library_embeddings' not in st.session_state:
    st.session_state.library_embeddings = None
if 'mitre_embeddings' not in st.session_state:
    st.session_state.mitre_embeddings = None
if '_uploaded_file' not in st.session_state:
    st.session_state._uploaded_file = None

# Initialize Claude Haiku Client
@st.cache_resource
def initialize_claude_client():
    try:
        # Get API key from environment variable or Streamlit secrets
        api_key = os.environ.get('ANTHROPIC_API_KEY') or st.secrets.get("ANTHROPIC_API_KEY", "")
        
        if not api_key:
            st.warning("Anthropic API key not found. Please set ANTHROPIC_API_KEY in environment variables or Streamlit secrets.")
            return None
        
        # Initialize the Claude client
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Claude client: {e}")
        return None

# Function to get MITRE mapping from Claude Haiku API
def get_claude_mitre_mapping(description, mitre_techniques, client):
    """
    Use Claude Haiku to map a description to MITRE ATT&CK techniques
    """
    if not client or not description:
        return "N/A", "N/A", "N/A", [], 0.0
    
    # Prepare techniques data for prompt
    techniques_summary = "\n".join([
        f"- {tech['id']} - {tech['name']}: {tech['tactic']}" 
        for tech in mitre_techniques[:20]  # Limit to 20 example techniques
    ])
    
    # Create a system prompt for Claude
    system_prompt = f"""
    You are a security analyst specialized in mapping security use cases to the MITRE ATT&CK framework.
    Given a security use case description, you will identify the most relevant MITRE ATT&CK technique that applies to it.
    Analyze the text carefully and match it to the most appropriate technique.
    
    Your response must be a JSON object with the following format:
    {{
        "tactic": "name of the MITRE tactic",
        "technique_name": "name of the technique without ID",
        "technique_id": "ID of the technique (e.g., T1078)",
        "confidence": a number between 0 and 100 indicating your confidence level,
        "explanation": "brief explanation of why this technique matches"
    }}
    
    Here are some example MITRE ATT&CK techniques for reference:
    {techniques_summary}
    """
    
    try:
        # Call Claude Haiku API
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Map this security use case to the MITRE ATT&CK framework: \n\n{description}"}
            ],
            temperature=0.2
        )
        
        # Extract JSON from the response
        response_text = message.content[0].text
        
        # Find JSON in the response (handling potential formatting issues)
        import re
        import json
        
        # Try to find JSON object in the response
        json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                
                # Extract values from the response
                tactic = result.get('tactic', 'N/A')
                technique_name = result.get('technique_name', 'N/A')
                technique_id = result.get('technique_id', 'N/A')
                confidence = result.get('confidence', 0)
                
                # Find the URL for the technique
                technique_url = "N/A"
                tactics_list = []
                
                # Look up the technique in our list to get its URL and tactics list
                for tech in mitre_techniques:
                    if tech['id'] == technique_id or tech['name'].lower() == technique_name.lower():
                        technique_url = tech['url']
                        tactics_list = tech['tactics_list']
                        break
                
                return tactic, technique_name, technique_url, tactics_list, confidence / 100.0
            except json.JSONDecodeError:
                # If JSON parsing fails, return a default response
                return "Parsing Error", "Parsing Error", "N/A", [], 0.0
        else:
            return "Response Error", "Response Error", "N/A", [], 0.0
    
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return "API Error", str(e)[:50], "N/A", [], 0.0

# Function to batch process mapping to MITRE with Claude Haiku
def batch_claude_mapping(descriptions: List[str], 
                         mitre_techniques: List[Dict],
                         claude_client) -> List[Tuple]:
    """
    Map a batch of descriptions to MITRE ATT&CK techniques using Claude Haiku API
    """
    if not claude_client:
        return [("N/A", "N/A", "N/A", [], 0.0) for _ in descriptions]
    
    results = []
    
    # Process each description individually with Claude
    for desc in descriptions:
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            results.append(("N/A", "N/A", "N/A", [], 0.0))
            continue
            
        # Clean the description
        clean_desc = str(desc).strip()
        if not clean_desc:
            results.append(("N/A", "N/A", "N/A", [], 0.0))
            continue
        
        # Get mapping from Claude Haiku
        tactic, technique, url, tactics_list, confidence = get_claude_mitre_mapping(
            clean_desc, mitre_techniques, claude_client
        )
        
        results.append((tactic, technique, url, tactics_list, confidence))
    
    return results

# Optimized function to check for library matches in batches
def batch_check_library_matches(descriptions: List[str], 
                              library_df: pd.DataFrame,
                              library_embeddings: torch.Tensor,
                              _model: SentenceTransformer,
                              batch_size: int = 32,
                              similarity_threshold: float = 0.8) -> List[Tuple]:
    """
    Check for matches in the library in batches for better performance.
    Returns a list of tuples: (matched_row, score, match_message)
    """
    if library_df is None or library_df.empty or library_embeddings is None:
        return [(None, 0.0, "No library data available") for _ in descriptions]
    
    results = []
    
    # First try exact matches (fast text comparison)
    exact_matches = {}
    for i, desc in enumerate(descriptions):
        # Handle NaN, None or float values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            exact_matches[i] = (None, 0.0, "Invalid description (None or numeric value)")
            continue
            
        # Convert to lowercase for case-insensitive matching
        try:
            lower_desc = str(desc).lower()
            
            # Check if there's an exact match in library
            matches = library_df[library_df['Description'].str.lower() == lower_desc]
            if not matches.empty:
                exact_matches[i] = (matches.iloc[0], 1.0, "Exact match found in library")
        except Exception as e:
            # Handle any errors in string operations
            exact_matches[i] = (None, 0.0, f"Error processing description: {str(e)}")
    
    # Process descriptions in batches for embeddings
    query_embeddings_list = []
    
    # Process only the descriptions that didn't have exact matches
    remaining_indices = [i for i in range(len(descriptions)) if i not in exact_matches]
    
    # Validate remaining descriptions for encoding
    valid_indices = []
    valid_descriptions = []
    
    for idx in remaining_indices:
        desc = descriptions[idx]
        # Skip None or non-string values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            results.append((idx, (None, 0.0, "Invalid description (None or numeric value)")))
        else:
            valid_indices.append(idx)
            valid_descriptions.append(str(desc))  # Convert to string to be safe
    
    # Skip if no valid descriptions remain
    if not valid_descriptions:
        return [exact_matches.get(i, (None, 0.0, "No match found in library")) for i in range(len(descriptions))]
    
    # Encode in batches
    for i in range(0, len(valid_descriptions), batch_size):
        batch = valid_descriptions[i:i+batch_size]
        try:
            batch_embeddings = _model.encode(batch, convert_to_tensor=True)
            
            # Perform search for this batch using PyTorch
            for j, query_embedding in enumerate(batch_embeddings):
                best_score, best_idx = cosine_similarity_search(query_embedding, library_embeddings)
                
                orig_idx = valid_indices[i + j]
                
                if best_score >= similarity_threshold:
                    results.append((orig_idx, (library_df.iloc[best_idx], best_score, 
                                f"Similar match found in library (score: {best_score:.2f})")))
                else:
                    results.append((orig_idx, (None, 0.0, "No match found in library")))
        except Exception as e:
            # Handle encoding errors
            for j in range(len(batch)):
                if i+j < len(valid_indices):
                    orig_idx = valid_indices[i + j]
                    results.append((orig_idx, (None, 0.0, f"Error during embedding: {str(e)}")))
    
    # Combine exact matches and embedding-based matches
    all_results = []
    for i in range(len(descriptions)):
        if i in exact_matches:
            all_results.append(exact_matches[i])
        else:
            # Find the result for this index
            result_found = False
            for idx, result in results:
                if idx == i:
                    all_results.append(result)
                    result_found = True
                    break
            if not result_found:
                all_results.append((None, 0.0, "No match found in library"))
    
    return all_results

# Main optimized mapping processing function
def process_mappings(df, _model, mitre_techniques, mitre_embeddings, library_df, library_embeddings):
    """
    Main function to process mappings in an optimized way
    
    This version uses:
    - all-mpnet-base-v2 for library matching
    - Claude Haiku for MITRE ATT&CK technique mapping
    """
    # Fixed similarity threshold
    similarity_threshold = 0.8
    
    # Get all descriptions at once and validate them
    descriptions = []
    for desc in df['Description'].tolist():
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            descriptions.append("No description available")
        else:
            descriptions.append(str(desc))  # Convert to string to ensure it's a string
    
    # First batch check library matches (using sentence-transformers model)
    library_match_results = batch_check_library_matches(
        descriptions, library_df, library_embeddings, _model, similarity_threshold=similarity_threshold
    )
    
    # Initialize Claude Haiku client
    claude_client = initialize_claude_client()
    
    # Prepare lists for rows that need model mapping
    model_map_indices = []
    model_map_descriptions = []
    
    # Process results and collect cases needing model mapping
    tactics = []
    techniques = []
    references = []
    all_tactics_lists = []
    confidence_scores = []
    match_sources = []
    match_scores = []
    techniques_count = {}
    
    # Make sure all lists have entries for each row in the dataframe
    for _ in range(len(df)):
        tactics.append("N/A")
        techniques.append("N/A")
        references.append("N/A")
        all_tactics_lists.append([])
        confidence_scores.append(0)
        match_sources.append("N/A")
        match_scores.append(0)
    
    for i, library_match in enumerate(library_match_results):
        matched_row, match_score, match_source = library_match
        
        if matched_row is not None:
            # Use library match
            tactic = matched_row.get('Mapped MITRE Tactic(s)', 'N/A')
            technique = matched_row.get('Mapped MITRE Technique(s)', 'N/A')
            reference = matched_row.get('Reference Resource(s)', 'N/A')
            
            tactics_list = tactic.split(', ') if tactic != 'N/A' else []
            confidence = match_score
            
            # Store results
            tactics[i] = tactic
            techniques[i] = technique
            references[i] = reference
            all_tactics_lists[i] = tactics_list
            confidence_scores[i] = round(confidence * 100, 2)
            match_sources[i] = match_source
            match_scores[i] = round(match_score * 100, 2)
            
            # Count techniques
            if technique != 'N/A':
                # Handle technique IDs in different formats
                if '-' in technique:
                    tech_id = technique.split('-')[0].strip()
                elif ' - ' in technique:
                    tech_id = technique.split(' - ')[0].strip()
                else:
                    # Try to find the technique by name in our list
                    tech_id = None
                    for tech in mitre_techniques:
                        if tech['name'].lower() == technique.lower():
                            tech_id = tech['id']
                            break
                    if not tech_id:
                        tech_id = technique  # Use technique name as fallback
                
                techniques_count[tech_id] = techniques_count.get(tech_id, 0) + 1
        else:
            # Make sure we're not trying to map invalid descriptions
            if not (descriptions[i] == "No description available" or pd.isna(descriptions[i])):
                # Mark for model mapping
                model_map_indices.append(i)
                model_map_descriptions.append(descriptions[i])
            else:
                # Invalid description placeholders are already set by default
                match_sources[i] = "Invalid description"
    
    # Batch map remaining cases using Claude Haiku
    if model_map_descriptions:
        # Progress indicator for Claude API calls
        claude_progress = st.progress(0)
        st.write("Using Claude Haiku for remaining mappings...")
        
        model_results = []
        
        # Process in smaller batches to update progress
        batch_size = 5  # Smaller batch size to show progress more frequently
        total_batches = (len(model_map_descriptions) + batch_size - 1) // batch_size
        
        for b in range(total_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(model_map_descriptions))
            
            batch_descriptions = model_map_descriptions[start_idx:end_idx]
            batch_results = batch_claude_mapping(
                batch_descriptions, mitre_techniques, claude_client
            )
            
            model_results.extend(batch_results)
            
            # Update progress
            progress = (b + 1) / total_batches
            claude_progress.progress(progress)
        
        # Process Claude results and insert at the correct positions
        for (i, idx) in enumerate(model_map_indices):
            if i < len(model_results):
                tactic, technique_name, reference, tactics_list, confidence = model_results[i]
                
                # Store the results
                tactics[idx] = tactic
                techniques[idx] = technique_name
                references[idx] = reference
                all_tactics_lists[idx] = tactics_list
                confidence_scores[idx] = round(confidence * 100, 2)
                match_sources[idx] = "Claude AI mapping"
                match_scores[idx] = 0  # No library match score
                
                # Count techniques - find technique ID if possible
                found_id = None
                for tech in mitre_techniques:
                    if tech['name'].lower() == technique_name.lower():
                        found_id = tech['id']
                        break
                
                tech_id = found_id or technique_name
                if tech_id != "N/A" and tech_id != "Error":
                    techniques_count[tech_id] = techniques_count.get(tech_id, 0) + 1
    
    # Add results to dataframe
    df['Mapped MITRE Tactic(s)'] = tactics
    df['Mapped MITRE Technique(s)'] = techniques
    df['Reference Resource(s)'] = references
    df['Confidence Score (%)'] = confidence_scores
    df['Match Source'] = match_sources
    df['Library Match Score (%)'] = match_scores
    
    return df, techniques_count

# Load embedding model with error handling
@st.cache_resource
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Using all-mpnet-base-v2 model
        model = SentenceTransformer('all-mpnet-base-v2')
        model = model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_mitre_data():
    try:
        response = requests.get("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json")
        attack_data = response.json()
        techniques = []
        tactic_mapping = {}
        tactics_list = []

        for obj in attack_data['objects']:
            if obj.get('type') == 'x-mitre-tactic':
                tactic_id = obj.get('external_references', [{}])[0].get('external_id', 'N/A')
                tactic_name = obj.get('name', 'N/A')
                tactic_mapping[tactic_name] = tactic_id
                tactics_list.append(tactic_name)

        for obj in attack_data['objects']:
            if obj.get('type') == 'attack-pattern':
                tech_id = obj.get('external_references', [{}])[0].get('external_id', 'N/A')
                if '.' in tech_id:
                    continue  # Skip sub-techniques
                techniques.append({
                    'id': tech_id,
                    'name': obj.get('name', 'N/A'),
                    'description': obj.get('description', ''),
                    'tactic': ', '.join([phase['phase_name'] for phase in obj.get('kill_chain_phases', [])]),
                    'tactics_list': [phase['phase_name'] for phase in obj.get('kill_chain_phases', [])],
                    'url': obj.get('external_references', [{}])[0].get('url', '')
                })
        
        return techniques, tactic_mapping, tactics_list
    except Exception as e:
        st.error(f"Error loading MITRE data: {e}")
        return [], {}, []

# Optimize the MITRE embeddings function for PyTorch
@st.cache_resource
def get_mitre_embeddings(_model, techniques):
    if _model is None or not techniques:
        return None
    try:
        descriptions = [tech['description'] for tech in techniques]
        
        # Encode all descriptions in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i+batch_size]
            batch_embeddings = _model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {e}")
        return None

# Optimized function to load and cache library data with embeddings
@st.cache_data
def load_library_data_with_embeddings(_model):
    try:
        # Read library.csv file
        try:
            library_df = pd.read_csv("library.csv")
        except:
            st.warning("Could not load library.csv file. Starting with an empty library.")
            # Create an empty DataFrame with required columns
            library_df = pd.DataFrame(columns=['Use Case Name', 'Description', 'Log Source', 
                                               'Mapped MITRE Tactic(s)', 'Mapped MITRE Technique(s)', 
                                               'Reference Resource(s)', 'Search'])
        
        if library_df.empty:
            return None, None
        
        # Fill NaN values with placeholders
        for col in library_df.columns:
            if library_df[col].dtype == 'object':
                library_df[col] = library_df[col].fillna("N/A")
        
        # Precompute embeddings for all library entries
        descriptions = []
        for desc in library_df['Description'].tolist():
            if pd.isna(desc) or isinstance(desc, float):
                descriptions.append("No description available")  # Safe fallback
            else:
                descriptions.append(str(desc))  # Ensure it's a string
        
        # Use batching for encoding
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i+batch_size]
            batch_embeddings = _model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            return library_df, embeddings
        
        return library_df, None
        
    except Exception as e:
        st.warning(f"Warning: Could not load library data: {e}")
        return None, None

def create_navigator_layer(techniques_count):
    try:
        techniques_data = []
        for tech_id, count in techniques_count.items():
            techniques_data.append({
                "techniqueID": tech_id,
                "score": count,
                "color": "",
                "comment": f"Count: {count}",
                "enabled": True,
                "metadata": [],
                "links": [],
                "showSubtechniques": False
            })
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        layer_id = str(uuid.uuid4())
        
        layer = {
            "name": f"Security Use Cases Mapping - {current_date}",
            "versions": {
                "attack": "17",
                "navigator": "4.8.1",
                "layer": "4.4"
            },
            "domain": "enterprise-attack",
            "description": f"Mapping of security use cases to MITRE ATT&CK techniques, generated on {current_date}",
            "filters": {
                "platforms": ["Linux", "macOS", "Windows", "Network", "PRE", "Containers", "Office 365", "SaaS", "IaaS", "Google Workspace", "Azure AD"]
            },
            "sorting": 0,
            "layout": {
                "layout": "side",
                "aggregateFunction": "max",
                "showID": True,
                "showName": True,
                "showAggregateScores": True,
                "countUnscored": False
            },
            "hideDisabled": False,
            "techniques": techniques_data,
            "gradient": {
                "colors": ["#ffffff", "#66b1ff", "#0d4a90"],
                "minValue": 0,
                "maxValue": max(techniques_count.values()) if techniques_count else 1
            },
            "legendItems": [],
            "metadata": [],
            "links": [],
            "showTacticRowBackground": True,
            "tacticRowBackground": "#dddddd",
            "selectTechniquesAcrossTactics": True,
            "selectSubtechniquesWithParent": False
        }
        
        return json.dumps(layer, indent=2), layer_id
    except Exception as e:
        st.error(f"Error creating Navigator layer: {e}")
        return "{}", ""

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Main application flow
def main():
    # Load the ML model and MITRE data
    model = load_model()
    mitre_techniques, tactic_mapping, tactics_list = load_mitre_data()

    # Load MITRE embeddings (for library matching and suggestions)
    mitre_embeddings = get_mitre_embeddings(model, mitre_techniques)
    st.session_state.mitre_embeddings = mitre_embeddings

    # Load library data with optimized embedding search
    library_df, library_embeddings = load_library_data_with_embeddings(model)
    if library_df is not None:
        st.session_state.library_data = library_df
        st.session_state.library_embeddings = library_embeddings

    # Store model in session state for use in suggestions
    st.session_state.model = model

    # Sidebar navigation
    with st.sidebar:
        st.image("https://attack.mitre.org/theme/images/mitre_attack_logo.png", width=200)
        
        selected = option_menu(
            "Navigation",
            ["Home", "Results", "Analytics", "Suggestions", "Export"],
            icons=['house', 'table', 'graph-up', 'search', 'box-arrow-down'],
            menu_icon="list",
            default_index=0,
        )
        
        st.session_state.page = selected.lower()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool maps your security use cases to the MITRE ATT&CK framework using:
        
        1. Library matching for known use cases
        2. Claude Haiku AI for new use cases mapping
        3. Suggestions for additional use cases based on your log sources
        
        - Upload a CSV with security use cases
        - Get automatic MITRE ATT&CK mappings
        - View suggested additional use cases
        - Visualize your coverage
        - Export for MITRE Navigator
        """)
        
        st.markdown("---")
        st.markdown("© 2025 | v1.5.0 (Claude Haiku Integration)")

    # Home page
    if st.session_state.page == "home":
        render_home_page(model, mitre_techniques, library_df, library_embeddings)

    # Results page
    elif st.session_state.page == "results":
        render_results_page()

    # Analytics page - Using the imported function
    elif st.session_state.page == "analytics":
        render_analytics_page(mitre_techniques)

    # Suggestions page
    elif st.session_state.page == "suggestions":
        render_suggestions_page()

    # Export page
    elif st.session_state.page == "export":
        render_export_page()

def render_home_page(model, mitre_techniques, library_df, library_embeddings):
    st.markdown("# 🛡️ MITRE ATT&CK Mapping Tool")
    st.markdown("### Map your security use cases to the MITRE ATT&CK framework")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Upload Security Use Cases")
        
        # Add animation
        lottie_upload = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_F0tVCP.json")
        if lottie_upload:
            st_lottie(lottie_upload, height=200, key="upload_animation")
        
        st.markdown("Upload a CSV file containing your security use cases. The file should include the columns: 'Use Case Name', 'Description', and 'Log Source'.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Store the uploaded file in session state for later use in suggestions
                st.session_state._uploaded_file = uploaded_file
                
                # Check for required columns
                required_cols = ['Use Case Name', 'Description', 'Log Source']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Your CSV must contain the columns: {', '.join(required_cols)}")
                else:
                    st.session_state.file_uploaded = True
                    st.success(f"File uploaded successfully! {len(df)} security use cases found.")
                    
                    # Fill NaN values with placeholder text for all important columns
                    for col in df.columns:
                        if df[col].dtype == 'object' or col in required_cols:
                            df[col] = df[col].fillna("N/A")
                    
                    st.markdown("""
                    1. **Upload** your security use cases CSV file
                    2. The tool first **checks** if the use case exists in the library
                    3. If found in library, it uses the **pre-mapped** MITRE data
                    4. If not found, it uses **Claude Haiku AI** to analyze and map the use case
                    5. **View** mapped results, analytics, and export options
                    6. **Discover** additional relevant use cases based on your log sources
                    """)
                    
                    # Show preview of the uploaded data
                    st.markdown("### Preview of Uploaded Data")
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # API key input for Claude Haiku
                    api_key = st.text_input(
                        "Enter Anthropic API Key (required for Claude Haiku mapping)",
                        type="password",
                        help="Required to use Claude Haiku for mapping. This will be stored temporarily for this session only."
                    )
                    
                    # Store API key in environment variable
                    if api_key:
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                    
                    # Show library statistics if available
                    if st.session_state.library_data is not None:
                        st.info(f"Library has {len(st.session_state.library_data)} pre-mapped security use cases that will be matched first.")
                    
                    map_button_disabled = not api_key
                    if map_button_disabled:
                        st.warning("Please enter your Anthropic API Key to enable mapping with Claude Haiku.")
                    
                    if st.button("Start Mapping", key="start_mapping", disabled=map_button_disabled):
                        with st.spinner("Mapping security use cases to MITRE ATT&CK..."):
                            # Progress bar
                            progress_bar = st.progress(0)
                            start_time = time.time()
                            
                            try:
                                # Use the optimized batch processing function
                                df, techniques_count = process_mappings(
                                    df, 
                                    model, 
                                    mitre_techniques, 
                                    st.session_state.mitre_embeddings,
                                    st.session_state.library_data,
                                    st.session_state.library_embeddings
                                )
                                
                                # Store processed data in session state
                                st.session_state.processed_data = df
                                st.session_state.techniques_count = techniques_count
                                st.session_state.mapping_complete = True
                                
                                # Complete
                                elapsed_time = time.time() - start_time
                                progress_bar.progress(100)
                                
                                st.success(f"Mapping complete in {elapsed_time:.2f} seconds! Navigate to Results to view the data.")
                                
                                # Add a suggestion to check the new Suggestions page
                                st.info("Don't forget to check the Suggestions page for additional use cases based on your log sources!")
                                
                                # Add a button to go directly to suggestions
                                if st.button("View Suggested Use Cases"):
                                    st.session_state.page = "suggestions"
                                    st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error during mapping process: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
                                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
    with col2:
        st.markdown("### How It Works")
        
        with st.expander("📝 Requirements", expanded=True):
            st.markdown("""
            Your CSV file should include:
            - 'Use Case Name': Name of the security use case
            - 'Description': Detailed description of the use case
            - 'Log Source': The log source for the use case
            
            You'll also need:
            - An Anthropic API key for Claude Haiku
            """)
        
        with st.expander("🔄 Process", expanded=True):
            st.markdown("""
            1. **Upload** your security use cases CSV file
            2. The tool first **checks** if the use case exists in the library using all-mpnet-base-v2
            3. If found in library, it uses the **pre-mapped** MITRE data
            4. If not found, it uses **Claude Haiku AI** to analyze the use case
            5. **View** mapped results, analytics, and export options
            6. **Discover** additional relevant use cases based on your log sources
            """)
        
        with st.expander("🤖 AI Integration", expanded=True):
            st.markdown("""
            This tool leverages two AI models:
            
            - **all-mpnet-base-v2**: Used for library matching, suggestions, and other embedding tasks
            - **Claude Haiku**: Used for mapping use cases to MITRE ATT&CK techniques
            
            Claude Haiku provides more accurate and context-aware mapping than traditional embedding models.
            """)

def render_results_page():
    st.markdown("# 📊 Mapping Results")
    
    if st.session_state.mapping_complete and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        st.markdown("### Filtered Results")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Mapped MITRE Tactic(s)' in df.columns:
                # Handle potential NaN values by filling with N/A first
                tactics_series = df['Mapped MITRE Tactic(s)'].fillna("N/A")
                all_tactics = set()
                for tactic_str in tactics_series:
                    if isinstance(tactic_str, str):
                        for tactic in tactic_str.split(', '):
                            if tactic and tactic != 'N/A':
                                all_tactics.add(tactic)
                selected_tactics = st.multiselect("Filter by Tactics", options=sorted(list(all_tactics)), default=[])
        
        with col2:
            search_term = st.text_input("Search in Descriptions", "")
        
        with col3:
            # Add a filter for match source (library or model)
            if 'Match Source' in df.columns:
                # Fill NaN values for safe filtering
                match_sources = df['Match Source'].fillna("Unknown").unique()
                selected_sources = st.multiselect("Filter by Match Source", options=match_sources, default=[])
        
        # Apply filters - safe handling for all filters
        filtered_df = df.copy()
        
        if selected_tactics:
            # Safe filtering that handles NaN values
            mask = filtered_df['Mapped MITRE Tactic(s)'].fillna('').apply(
                lambda x: isinstance(x, str) and any(tactic in x for tactic in selected_tactics)
            )
            filtered_df = filtered_df[mask]
        
        if search_term:
            # Safe filtering that handles NaN values
            mask = filtered_df['Description'].fillna('').astype(str).str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        if selected_sources:
            # Safe filtering that handles NaN values
            mask = filtered_df['Match Source'].fillna('Unknown').astype(str).apply(
                lambda x: any(source in x for source in selected_sources)
            )
            filtered_df = filtered_df[mask]
        
        # Display results
        st.markdown(f"Showing {len(filtered_df)} of {len(df)} use cases")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download options
        st.download_button(
            "Download Results as CSV",
            filtered_df.to_csv(index=False).encode('utf-8'),
            "mitre_mapped_results.csv",
            "text/csv"
        )
    
    else:
        st.info("No mapping results available. Please upload a CSV file on the Home page and complete the mapping process.")
        
        # Add a button to navigate back to home
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()

def render_export_page():
    st.markdown("# 💾 Export Navigator Layer")
    
    if st.session_state.mapping_complete and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        st.markdown("### MITRE ATT&CK Navigator Export")
        
        navigator_layer, layer_id = create_navigator_layer(st.session_state.techniques_count)
        
        st.markdown("""
        The MITRE ATT&CK Navigator is an interactive visualization tool for exploring the MITRE ATT&CK framework.
        
        You can export your mapping results as a layer file to visualize in the Navigator.
        """)
        
        st.download_button(
            label="Download Navigator Layer JSON",
            data=navigator_layer,
            file_name="navigator_layer.json",
            mime="application/json",
            key="download_nav"
        )
        
        st.markdown("### How to Use in MITRE ATT&CK Navigator")
        
        st.markdown("""
        1. Download the Navigator Layer JSON using the button above
        2. Visit the [MITRE ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/)
        3. Click "Open Existing Layer" and then "Upload from Local"
        4. Select the downloaded `navigator_layer.json` file
        """)
        
        with st.expander("View Navigator Layer JSON"):
            st.code(navigator_layer, language="json")
    
    else:
        st.info("No export data available. Please upload a CSV file on the Home page and complete the mapping process.")
        
        # Add a button to navigate back to home
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()

if __name__ == '__main__':
    main()
