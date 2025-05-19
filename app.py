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
import anthropic  # Added for Claude Sonnet API

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

# Initialize Claude Sonnet Client
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

# Function to get MITRE mapping from Claude Sonnet API with Query focus
def get_claude_mitre_mapping(query, mitre_techniques, client):
    """
    Use Claude Sonnet to map a security query to MITRE ATT&CK techniques
    with improved accuracy and validation
    """
    if not client or not query or query == "N/A":
        return "N/A", "N/A", "N/A", [], 0.0
    
    # Create a lookup dictionary for technique validation
    technique_dict = {}
    technique_to_tactics = {}
    
    for tech in mitre_techniques:
        # Store by ID
        technique_dict[tech['id']] = tech
        
        # Store by name (lowercase for case-insensitive matching)
        technique_dict[tech['name'].lower()] = tech
        
        # Create mapping of technique name to valid tactics
        technique_to_tactics[tech['name'].lower()] = tech['tactics_list']
    
    # Prepare techniques data for prompt - include all techniques for better context
    techniques_summary = "\n".join([
        f"- {tech['id']} - {tech['name']}: {tech['tactic']}" 
        for tech in mitre_techniques[:50]  # Include more examples for better context
    ])
    
    # Create a more detailed system prompt for Claude with specific examples
    system_prompt = f"""
    You are a security analyst specialized in mapping security use cases to the MITRE ATT&CK framework.
    Given a security use case Query, you will identify the most relevant MITRE ATT&CK technique that applies to it.
    
    IMPORTANT RULES:
    1. Use ONLY techniques and tactics that exist in the MITRE ATT&CK framework
    2. Be precise with the tactic name - it must be one of the standard MITRE tactics (Initial Access, Execution, Persistence, etc.)
    3. For each technique, verify that the tactic you assign is valid for that technique, as techniques can appear in multiple tactics
    4. Always check that you're using the correct tactic for the context of the use case
    5. Focus on what the query is trying to detect rather than the query syntax itself
    
    EXAMPLES:
    - For queries detecting failed login attempts, use "Brute Force" technique with the "Credential Access" tactic
    - For queries looking for process creation, use "User Execution" technique with the "Execution" tactic
    - For queries detecting file modifications in system directories, use "Modify Registry" or related technique with "Defense Evasion" tactic
    - For queries monitoring network connections to suspicious IPs, use "Command and Control" tactic with appropriate technique
    
    Your response must be a JSON object with the following format:
    {{
        "tactic": "name of the MITRE tactic",
        "technique_name": "name of the technique without ID",
        "technique_id": "ID of the technique (e.g., T1078)",
        "confidence": a number between 0 and 100 indicating your confidence level,
        "explanation": "brief explanation of why this technique matches the security query"
    }}
    
    Here are MITRE ATT&CK techniques for reference:
    {techniques_summary}
    """
    
    try:
        # Call Claude Sonnet API
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Using Sonnet instead of Haiku
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Map this security query to the MITRE ATT&CK framework: \n\n{query}"}
            ],
            temperature=0.1  # Lower temperature for more deterministic results
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
                
                # Validation step: Check if the technique exists and the tactic is valid for this technique
                valid_mapping = False
                correct_tactic = tactic
                correct_technique = technique_name
                
                # Try to find the technique in our reference data
                if technique_id in technique_dict:
                    tech = technique_dict[technique_id]
                    valid_tactics = tech['tactics_list']
                    
                    # Check if the assigned tactic is valid for this technique
                    if tactic in valid_tactics:
                        valid_mapping = True
                        correct_technique = tech['name']  # Use the official name
                    else:
                        # If the tactic is invalid, use the first valid tactic
                        correct_tactic = valid_tactics[0] if valid_tactics else tactic
                        correct_technique = tech['name']
                
                # If technique wasn't found by ID, try by name
                elif technique_name.lower() in technique_dict:
                    tech = technique_dict[technique_name.lower()]
                    valid_tactics = tech['tactics_list']
                    technique_id = tech['id']  # Update the ID
                    
                    # Check if the assigned tactic is valid for this technique
                    if tactic in valid_tactics:
                        valid_mapping = True
                        correct_technique = tech['name']  # Use the official name
                    else:
                        # If the tactic is invalid, use the first valid tactic
                        correct_tactic = valid_tactics[0] if valid_tactics else tactic
                        correct_technique = tech['name']
                
                # Find the URL for the technique
                technique_url = "N/A"
                tactics_list = []
                
                # Look up the technique in our list to get its URL and tactics list
                for tech in mitre_techniques:
                    if tech['id'] == technique_id or tech['name'].lower() == technique_name.lower():
                        technique_url = tech['url']
                        tactics_list = tech['tactics_list']
                        break
                
                # Return the correct mapping (either validated or corrected)
                if valid_mapping:
                    return tactic, correct_technique, technique_url, tactics_list, confidence / 100.0
                else:
                    return correct_tactic, correct_technique, technique_url, tactics_list, (confidence * 0.9) / 100.0  # Reduce confidence if we had to correct
            except json.JSONDecodeError:
                # If JSON parsing fails, return a default response
                return "Parsing Error", "Parsing Error", "N/A", [], 0.0
        else:
            return "Response Error", "Response Error", "N/A", [], 0.0
    
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return "API Error", str(e)[:50], "N/A", [], 0.0

# Function to batch process mapping to MITRE with Claude Sonnet (Query focused)
def batch_claude_mapping(queries: List[str],
                         mitre_techniques: List[Dict],
                         claude_client) -> List[Tuple]:
    """
    Map a batch of queries to MITRE ATT&CK techniques using Claude Sonnet API
    """
    if not claude_client:
        return [("N/A", "N/A", "N/A", [], 0.0) for _ in queries]
    
    results = []
    
    # Process each query individually with Claude
    for query in queries:
        if pd.isna(query) or query is None or isinstance(query, float) or query == "N/A":
            results.append(("N/A", "N/A", "N/A", [], 0.0))
            continue
            
        # Clean the query
        clean_query = str(query).strip()
        if not clean_query:
            results.append(("N/A", "N/A", "N/A", [], 0.0))
            continue
        
        # Get mapping from Claude Sonnet
        tactic, technique, url, tactics_list, confidence = get_claude_mitre_mapping(
            clean_query, mitre_techniques, claude_client
        )
        
        results.append((tactic, technique, url, tactics_list, confidence))
    
    return results

# Optimized function to check for library matches in batches (Updated for Query support)
def batch_check_library_matches(descriptions: List[str], 
                              queries: List[str],
                              library_df: pd.DataFrame,
                              library_embeddings: torch.Tensor,
                              _model: SentenceTransformer,
                              batch_size: int = 32,
                              similarity_threshold: float = 0.8) -> List[Tuple]:
    """
    Check for matches in the library in batches for better performance.
    Now includes query information in the matching process.
    Returns a list of tuples: (matched_row, score, match_message)
    """
    if library_df is None or library_df.empty or library_embeddings is None:
        return [(None, 0.0, "No library data available") for _ in descriptions]
    
    results = []
    
    # First try exact matches (fast text comparison)
    exact_matches = {}
    for i, desc in enumerate(descriptions):
        query = queries[i] if i < len(queries) else ""
        
        # Handle NaN, None or float values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            desc = ""
        if pd.isna(query) or query is None or isinstance(query, float):
            query = ""
            
        if not desc and not query:
            exact_matches[i] = (None, 0.0, "Invalid description and query (both empty)")
            continue
            
        # Convert to lowercase for case-insensitive matching
        try:
            # Combine description and query for matching
            combined_text = ""
            if desc:
                combined_text += str(desc).lower()
            if query:
                if combined_text:
                    combined_text += " " + str(query).lower()
                else:
                    combined_text = str(query).lower()
            
            # Check if there's an exact match in library (checking both description and query if available)
            for _, lib_row in library_df.iterrows():
                lib_desc = str(lib_row.get('Description', '')).lower()
                lib_query = str(lib_row.get('Query', '')).lower() if 'Query' in lib_row else ""
                
                # Check exact match for description
                if desc and desc.lower() == lib_desc:
                    exact_matches[i] = (lib_row, 1.0, "Exact description match found in library")
                    break
                # Check exact match for query if available
                elif query and lib_query and query.lower() == lib_query:
                    exact_matches[i] = (lib_row, 1.0, "Exact query match found in library")
                    break
        except Exception as e:
            # Handle any errors in string operations
            exact_matches[i] = (None, 0.0, f"Error processing description/query: {str(e)}")
    
    # Process descriptions in batches for embeddings
    query_embeddings_list = []
    
    # Process only the cases that didn't have exact matches
    remaining_indices = [i for i in range(len(descriptions)) if i not in exact_matches]
    
    # Validate remaining descriptions for encoding
    valid_indices = []
    valid_texts = []
    
    for idx in remaining_indices:
        desc = descriptions[idx]
        query = queries[idx] if idx < len(queries) else ""
        
        # Handle None or non-string values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            desc = ""
        if pd.isna(query) or query is None or isinstance(query, float):
            query = ""
            
        # Combine description and query for embedding
        combined_text = ""
        if desc:
            combined_text += str(desc)
        if query:
            if combined_text:
                combined_text += " " + str(query)
            else:
                combined_text = str(query)
        
        if combined_text.strip():
            valid_indices.append(idx)
            valid_texts.append(combined_text)
        else:
            results.append((idx, (None, 0.0, "No valid description or query available")))
    
    # Skip if no valid texts remain
    if not valid_texts:
        return [exact_matches.get(i, (None, 0.0, "No match found in library")) for i in range(len(descriptions))]
    
    # Encode in batches
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i+batch_size]
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

# Main optimized mapping processing function (Updated for Query support)
def process_mappings(df, _model, mitre_techniques, mitre_embeddings, library_df, library_embeddings):
    """
    Main function to process mappings in an optimized way
    
    This version uses:
    - all-mpnet-base-v2 for library matching
    - Claude Sonnet for MITRE ATT&CK technique mapping with validation
    - Now supports Query column for enhanced mapping context, but mapping is done based on Query only
    """
    # Fixed similarity threshold
    similarity_threshold = 0.8
    
    # Create lookup dictionaries for faster validation
    technique_id_to_name = {}
    technique_name_to_id = {}
    technique_to_tactics = {}
    
    for tech in mitre_techniques:
        technique_id_to_name[tech['id']] = tech['name']
        technique_name_to_id[tech['name'].lower()] = tech['id']
        technique_to_tactics[tech['name'].lower()] = set(tech['tactics_list'])
    
    # Get all descriptions and queries at once and validate them
    descriptions = []
    queries = []
    
    for desc in df['Description'].tolist():
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            descriptions.append("No description available")
        else:
            descriptions.append(str(desc))
    
    # Handle Query column if it exists
    if 'Query' in df.columns:
        for query in df['Query'].tolist():
            if pd.isna(query) or query is None or isinstance(query, float):
                queries.append("")
            else:
                queries.append(str(query))
    else:
        queries = [""] * len(descriptions)
    
    # First batch check library matches (using sentence-transformers model with query support)
    library_match_results = batch_check_library_matches(
        descriptions, queries, library_df, library_embeddings, _model, similarity_threshold=similarity_threshold
    )
    
    # Initialize Claude Sonnet client
    claude_client = initialize_claude_client()
    
    # Prepare lists for rows that need model mapping
    model_map_indices = []
    model_map_queries = []
    
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
            
            # Validate the tactic-technique relationship
            validated_tactic = tactic
            validated_technique = technique
            
            # Check if the technique is in our reference data
            if technique != 'N/A':
                # Extract the technique name (removing ID if present)
                if ' - ' in technique:
                    tech_name = technique.split(' - ')[1].strip().lower()
                elif '-' in technique and technique[0] == 'T':
                    tech_id = technique.split('-')[0].strip()
                    if tech_id in technique_id_to_name:
                        tech_name = technique_id_to_name[tech_id].lower()
                    else:
                        tech_name = technique.lower()
                else:
                    tech_name = technique.lower()
                
                # Check if this technique has valid tactics and if the assigned tactic is valid
                if tech_name in technique_to_tactics:
                    valid_tactics = technique_to_tactics[tech_name]
                    
                    # If tactic is a comma-separated list, check each tactic
                    if ',' in tactic:
                        tactic_list = [t.strip() for t in tactic.split(',')]
                        validated_tactics = []
                        
                        for t in tactic_list:
                            if t in valid_tactics:
                                validated_tactics.append(t)
                        
                        if validated_tactics:
                            validated_tactic = ', '.join(validated_tactics)
                    else:
                        # Single tactic case
                        if tactic not in valid_tactics and valid_tactics:
                            # If the tactic is invalid, use the first valid tactic
                            validated_tactic = list(valid_tactics)[0]
            
            tactics_list = validated_tactic.split(', ') if validated_tactic != 'N/A' else []
            confidence = match_score
            
            # Store results with validated data
            tactics[i] = validated_tactic
            techniques[i] = validated_technique
            references[i] = reference
            all_tactics_lists[i] = tactics_list
            confidence_scores[i] = round(confidence * 100, 2)
            match_sources[i] = match_source
            match_scores[i] = round(match_score * 100, 2)
            
            # Count techniques
            if validated_technique != 'N/A':
                # Extract the technique ID
                if ' - ' in validated_technique:
                    parts = validated_technique.split(' - ')
                    tech_id = parts[0].strip()
                    tech_name = parts[1].strip().lower()
                elif validated_technique.startswith('T') and len(validated_technique) >= 5:
                    # Looks like an ID
                    tech_id = validated_technique
                else:
                    # Assume it's a name and look it up
                    tech_name = validated_technique.lower()
                    if tech_name in technique_name_to_id:
                        tech_id = technique_name_to_id[tech_name]
                    else:
                        # Fall back to using the name as is
                        tech_id = validated_technique
                
                techniques_count[tech_id] = techniques_count.get(tech_id, 0) + 1
        else:
            # Check if we have valid query for mapping (only query-based mapping now)
            has_valid_query = i < len(queries) and queries[i] and not pd.isna(queries[i]) and queries[i].strip()
            
            if has_valid_query:
                # Mark for model mapping
                model_map_indices.append(i)
                model_map_queries.append(queries[i])
            else:
                # Invalid or missing query
                match_sources[i] = "Invalid or missing query"
    
    # Batch map remaining cases using Claude Sonnet (Query-based only)
    if model_map_queries:
        # Progress indicator for Claude API calls
        claude_progress = st.progress(0)
        st.write("Using Claude Sonnet for query-based mappings...")
        
        model_results = []
        
        # Process in smaller batches to update progress
        batch_size = 5  # Smaller batch size to show progress more frequently
        total_batches = (len(model_map_queries) + batch_size - 1) // batch_size
        
        for b in range(total_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(model_map_queries))
            
            batch_queries = model_map_queries[start_idx:end_idx]
            batch_results = batch_claude_mapping(
                batch_queries, mitre_techniques, claude_client
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
                match_sources[idx] = "Claude AI query mapping"
                match_scores[idx] = 0  # No library match score
                
                # Count techniques - find technique ID if possible
                if technique_name.lower() in technique_name_to_id:
                    tech_id = technique_name_to_id[technique_name.lower()]
                else:
                    # Try to find by name similarity
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

# Optimized function to load and cache library data with embeddings (Updated for Query support)
@st.cache_data
def load_library_data_with_embeddings(_model):
    try:
        # Read library.csv file
        try:
            library_df = pd.read_csv("library.csv")
        except:
            st.warning("Could not load library.csv file. Starting with an empty library.")
            # Create an empty DataFrame with required columns (now including Query)
            library_df = pd.DataFrame(columns=['Use Case Name', 'Description', 'Log Source', 
                                               'Mapped MITRE Tactic(s)', 'Mapped MITRE Technique(s)', 
                                               'Reference Resource(s)', 'Query'])
        
        if library_df.empty:
            return None, None
        
        # Fill NaN values with placeholders
        for col in library_df.columns:
            if library_df[col].dtype == 'object':
                library_df[col] = library_df[col].fillna("N/A")
        
        # Precompute embeddings for all library entries (combine description and query)
        combined_texts = []
        for _, row in library_df.iterrows():
            desc = row.get('Description', '')
            query = row.get('Query', '') if 'Query' in row else ''
            
            # Handle NaN values
            if pd.isna(desc) or isinstance(desc, float):
                desc = "No description available"
            if pd.isna(query) or isinstance(query, float):
                query = ""
            
            # Combine description and query for embedding
            combined_text = str(desc)
            if query:
                combined_text += " " + str(query)
            
            combined_texts.append(combined_text)
        
        # Use batching for encoding
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(combined_texts), batch_size):
            batch = combined_texts[i:i+batch_size]
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

# Create navigator layer function
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
        
        1. Library matching for known use cases and queries
        2. Claude Sonnet AI for new query/use case mapping
        3. Suggestions for additional use cases based on your log sources
        
        - Upload a CSV with security use cases (and optional queries)
        - Get automatic MITRE ATT&CK mappings based on queries or descriptions
        - View suggested additional use cases
        - Visualize your coverage
        - Export for MITRE Navigator
        """)
        
        st.markdown("---")
        st.markdown("© 2025 | v1.7.0 (Query-Focused Mapping)")

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
    st.markdown("### Map your security use cases and queries to the MITRE ATT&CK framework")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Upload Security Use Cases")
        
        # Add animation
        lottie_upload = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_F0tVCP.json")
        if lottie_upload:
            st_lottie(lottie_upload, height=200, key="upload_animation")
        
        st.markdown("Upload a CSV file containing your security use cases. The file should include the columns: 'Use Case Name', 'Description', 'Log Source', and 'Query'. The Query column will be used for MITRE ATT&CK mapping.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Store the uploaded file in session state for later use in suggestions
                st.session_state._uploaded_file = uploaded_file
                
                # Check for required columns - Query is optional but used for mapping
                required_cols = ['Use Case Name', 'Description', 'Log Source']
                optional_cols = ['Query']
                
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Your CSV must contain the columns: {', '.join(required_cols)}")
                    st.info(f"Optional column (used for mapping): {', '.join(optional_cols)}")
                else:
                    st.session_state.file_uploaded = True
                    
                    # Check if Query column exists
                    has_query_col = 'Query' in df.columns
                    query_info = f" (with Query column for enhanced mapping)" if has_query_col else " (Query column missing - mapping will use descriptions)"
                    
                    st.success(f"File uploaded successfully! {len(df)} security use cases found{query_info}.")
                    
                    # Fill NaN values with placeholder text for all important columns
                    for col in df.columns:
                        if df[col].dtype == 'object' or col in required_cols + optional_cols:
                            df[col] = df[col].fillna("N/A")
                    
                    st.markdown("""
                    1. **Upload** your security use cases CSV file
                    2. The tool first **checks** if the use case exists in the library
                    3. If found in library, it uses the **pre-mapped** MITRE data
                    4. If not found, it uses **Claude Sonnet AI** to analyze the security query (if available) or description
                    5. **View** mapped results, analytics, and export options
                    6. **Discover** additional relevant use cases based on your log sources
                    """)
                    
                    # Show preview of the uploaded data
                    st.markdown("### Preview of Uploaded Data")
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # API key input for Claude Sonnet
                    api_key = st.text_input(
                        "Enter Anthropic API Key (required for Claude Sonnet mapping)",
                        type="password",
                        help="Required to use Claude Sonnet for mapping. This will be stored temporarily for this session only."
                    )
                    
                    # Store API key in environment variable
                    if api_key:
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                    
                    # Show library statistics if available
                    if st.session_state.library_data is not None:
                        lib_has_query = 'Query' in st.session_state.library_data.columns
                        query_lib_info = " (including queries)" if lib_has_query else ""
                        st.info(f"Library has {len(st.session_state.library_data)} pre-mapped security use cases{query_lib_info} that will be matched first.")
                    
                    map_button_disabled = not api_key
                    if map_button_disabled:
                        st.warning("Please enter your Anthropic API Key to enable mapping with Claude Sonnet.")
                    
                    if st.button("Start Mapping", key="start_mapping", disabled=map_button_disabled):
                        with st.spinner("Mapping security use cases and queries to MITRE ATT&CK..."):
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
            - 'Query' (Optional but recommended): Security query/rule for the use case
            
            You'll also need:
            - An Anthropic API key for Claude Sonnet
            """)
        
        with st.expander("🔄 Process", expanded=True):
            st.markdown("""
            1. **Upload** your security use cases CSV file (with optional Query column)
            2. The tool first **checks** if the use case exists in the library using all-mpnet-base-v2
            3. If found in library, it uses the **pre-mapped** MITRE data
            4. If not found, it uses **Claude Sonnet AI** to analyze the security query (if available) or description
            5. **View** mapped results, analytics, and export options
            6. **Discover** additional relevant use cases based on your log sources
            """)
        
        with st.expander("🤖 AI Integration", expanded=True):
            st.markdown("""
            This tool leverages two AI models:
            
            - **all-mpnet-base-v2**: Used for library matching, suggestions, and other embedding tasks
            - **Claude Sonnet**: Used for mapping security queries (preferred) or descriptions to MITRE ATT&CK techniques
            
            Claude Sonnet provides accurate mapping by focusing on what security queries are designed to detect,
            making it ideal for mapping detection rules, SIEM queries, and other security logic to MITRE techniques.
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
            search_term = st.text_input("Search in Descriptions/Queries", "")
        
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
            # Safe filtering that handles NaN values - search in both Description and Query columns
            desc_mask = filtered_df['Description'].fillna('').astype(str).str.contains(search_term, case=False, na=False)
            
            # Check if Query column exists and search in it too
            if 'Query' in filtered_df.columns:
                query_mask = filtered_df['Query'].fillna('').astype(str).str.contains(search_term, case=False, na=False)
                mask = desc_mask | query_mask
            else:
                mask = desc_mask
            
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
