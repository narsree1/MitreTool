import pandas as pd
import streamlit as st
import datetime
import time
import json
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Import modules
from modules.data_loader import load_mitre_data, load_library_data_with_embeddings, create_local_mitre_data_cache
from modules.embedding import load_model, get_mitre_embeddings
from modules.mapper import process_mappings, get_suggested_use_cases
from modules.utils import load_lottie_url
from modules.visualizations import create_navigator_layer

# App configuration
st.set_page_config(
    page_title="MITRE ATT&CK Mapping Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('modules/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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
if 'model_name' not in st.session_state:
    st.session_state.model_name = "deepseek-ai/deepseek-sentence-embedding-v1"

# Render suggestions page
def render_suggestions_page():
    st.markdown("# 🔍 Suggested Use Cases")
    
    if st.session_state.file_uploaded:
        if st.session_state.library_data is not None and not st.session_state.library_data.empty:
            
            uploaded_df = None
            if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
                uploaded_df = st.session_state.processed_data
            else:
                # Try to get the original uploaded data if processing hasn't happened yet
                try:
                    uploaded_file = st.session_state.get('_uploaded_file')
                    if uploaded_file:
                        uploaded_df = pd.read_csv(uploaded_file)
                except:
                    pass
            
            if uploaded_df is None:
                st.info("Please upload your data file on the Home page first.")
                return
                
            # Get suggestions based on log sources
            with st.spinner("Finding suggested use cases based on log sources..."):
                log_source_suggestions = get_suggested_use_cases(
                    uploaded_df, 
                    st.session_state.library_data
                )
            
            # Display suggestions
            if not log_source_suggestions.empty:
                st.success(f"Found {len(log_source_suggestions)} suggested use cases based on your log sources!")
                
                # Format the dataframe for display
                display_df = log_source_suggestions.copy()
                if 'Relevance' in display_df.columns:
                    display_df['Relevance Score'] = display_df['Relevance'].apply(lambda x: f"{x:.0f} ⭐")
                    display_df = display_df.drop('Relevance', axis=1)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Add a detailed view for each suggestion
                st.markdown("### Detailed View")
                selected_suggestion = st.selectbox(
                    "Select a use case to view details",
                    options=display_df['Use Case Name'].tolist(),
                    index=0
                )
                
                if selected_suggestion:
                    selected_row = display_df[display_df['Use Case Name'] == selected_suggestion].iloc[0]
                    
                    # Create columns for the detailed view
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### Use Case Details")
                        st.markdown(f"**Name:** {selected_row.get('Use Case Name', 'N/A')}")
                        st.markdown(f"**Log Source:** {selected_row.get('Log Source', 'N/A')}")
                        st.markdown(f"**Description:**")
                        st.markdown(f"{selected_row.get('Description', 'No description available')}")
                    
                    with col2:
                        st.markdown("#### MITRE ATT&CK Mapping")
                        st.markdown(f"**Tactic(s):** {selected_row.get('Mapped MITRE Tactic(s)', 'N/A')}")
                        st.markdown(f"**Technique(s):** {selected_row.get('Mapped MITRE Technique(s)', 'N/A')}")
                        
                        # Display reference resources if available
                        if 'Reference Resource(s)' in selected_row and selected_row['Reference Resource(s)'] != 'N/A':
                            st.markdown("#### Reference Resources")
                            st.markdown(f"{selected_row['Reference Resource(s)']}")
                    
                    # Display search query in a separate section
                    if 'Search' in selected_row and selected_row['Search'] != 'N/A' and not pd.isna(selected_row['Search']):
                        st.markdown("### Search Query")
                        st.code(selected_row['Search'], language="sql")
                
                # Download option
                st.download_button(
                    "Download Suggested Use Cases as CSV",
                    log_source_suggestions.to_csv(index=False).encode('utf-8'),
                    "suggested_use_cases.csv",
                    "text/csv"
                )
            else:
                st.info("No additional use cases found based on your log sources.")
        else:
            st.warning("Library data is not available. Cannot provide suggestions without a reference library.")
    else:
        st.info("Please upload your security use cases CSV file on the Home page first.")
        
        # Add a button to navigate back to home
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()

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
    2. Natural language processing for new use cases
    3. Suggestions for additional use cases based on your log sources
    
    - Upload a CSV with security use cases
    - Get automatic MITRE ATT&CK mappings
    - View suggested additional use cases
    - Visualize your coverage
    - Export for MITRE Navigator
    """)
    
    st.markdown("---")
    st.markdown(f"© 2025 | v1.5.0 (DeepSeek Enhanced)")

# Load the ML model and MITRE data
model = load_model()
mitre_techniques, tactic_mapping, tactics_list = load_mitre_data()

# Load MITRE embeddings
mitre_embeddings = get_mitre_embeddings(model, mitre_techniques)
st.session_state.mitre_embeddings = mitre_embeddings

# Load library data with optimized embedding search
library_df, library_embeddings = load_library_data_with_embeddings(model)
if library_df is not None:
    st.session_state.library_data = library_df
    st.session_state.library_embeddings = library_embeddings

# Store model in session state for use in suggestions
st.session_state.model = model

# Home page
if st.session_state.page == "home":
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
        
        # Display model information
        st.info(f"Using DeepSeek embedding model for improved accuracy: {st.session_state.model_name}")
        
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
                    4. If not found, it **analyzes** the use case using NLP and maps it
                    5. **View** mapped results, analytics, and export options
                    6. **Discover** additional relevant use cases based on your log sources
                    """)
                    
                    # Show preview of the uploaded data
                    st.markdown("### Preview of Uploaded Data")
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # Show library statistics if available
                    if st.session_state.library_data is not None:
                        st.info(f"Library has {len(st.session_state.library_data)} pre-mapped security use cases that will be matched first.")
                    
                    if st.button("Start Mapping", key="start_mapping"):
                        with st.spinner("Mapping security use cases to MITRE ATT&CK using DeepSeek embeddings..."):
                            # Progress bar
                            progress_bar = st.progress(0)
                            start_time = time.time()
                            
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
                st.error(f"Error processing file: {str(e)}")
        
    with col2:
        st.markdown("### How It Works")
        
        with st.expander("📝 Requirements", expanded=True):
            st.markdown("""
            Your CSV file should include:
            - 'Use Case Name': Name of the security use case
            - 'Description': Detailed description of the use case
            - 'Log Source': The log source for the use case
            """)
        
        with st.expander("🔄 Process", expanded=True):
            st.markdown("""
            1. **Upload** your security use cases CSV file
            2. The tool first **checks** if the use case exists in the library
            3. If found in library, it uses the **pre-mapped** MITRE data
            4. If not found, it **analyzes** the use case using DeepSeek embeddings and maps it
            5. **View** mapped results, analytics, and export options
            6. **Discover** additional relevant use cases based on your log sources
            """)
        
        with st.expander("🚀 Improvements with DeepSeek", expanded=True):
            st.markdown("""
            This version uses DeepSeek embeddings for improved mapping accuracy:
            
            - **Better semantic understanding** of security terminology
            - **Higher quality mappings** for techniques and tactics
            - **Improved contextual awareness** for more accurate mapping
            - **Enhanced similarity detection** between use cases and MITRE techniques
            """)

# Results page
elif st.session_state.page == "results":
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

# Analytics, Suggestions, and Export pages remain the same...
