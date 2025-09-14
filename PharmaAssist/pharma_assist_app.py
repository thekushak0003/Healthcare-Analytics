# ------------------
# Imports
# ------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------
# Page Configuration
# ------------------
# Set the title and icon that appear in the browser tab.
st.set_page_config(
    page_title="PharmaAssist Recommender",
    page_icon="💊",
    layout="wide"
)

# ------------------
# Data Loading and Model Preparation (Cached for Performance)
# ------------------
# This function loads and processes the data. @st.cache_data ensures this only runs once.
@st.cache_data
def load_data_and_prepare_model():
    """
    Loads the medicine data, performs cleaning and feature engineering,
    and prepares the TF-IDF matrix for the recommendation model.
    """
    try:
        df = pd.read_parquet('medicine_data.parquet')
    except FileNotFoundError:
        st.error("Error: 'medicine_data.parquet' not found. Please make sure the file is in the same directory.")
        return None, None, None

    # --- Data Cleaning ---
    def get_cols_by_prefix(df, prefix):
        return [col for col in df.columns if col.startswith(prefix)]

    side_effect_cols = get_cols_by_prefix(df, 'sideEffect')
    use_cols = get_cols_by_prefix(df, 'use')
    substitute_cols = get_cols_by_prefix(df, 'substitute')

    df['all_side_effects'] = df[side_effect_cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)
    df['all_uses'] = df[use_cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)
    df['all_substitutes'] = df[substitute_cols].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

    df.drop(columns=side_effect_cols + use_cols + substitute_cols, inplace=True)

    df.dropna(subset=['Therapeutic Class'], inplace=True)
    fill_values = {
        'all_uses': 'Not specified',
        'all_side_effects': 'None reported',
        'all_substitutes': 'None available',
        'Habit Forming': 'Unknown'
    }
    df.fillna(value=fill_values, inplace=True)
    df.dropna(subset=['name'], inplace=True)
    df.rename(columns={'name': 'drug_name'}, inplace=True)
    df = df.reset_index(drop=True)

    # --- Feature Engineering & Vectorization ---
    features = ['Therapeutic Class', 'Action Class', 'Chemical Class', 'all_side_effects', 'all_uses']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['soup'] = df.apply(lambda x: ' '.join([x[f] for f in features]), axis=1)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    indices = pd.Series(df.index, index=df['drug_name']).drop_duplicates()

    return df, tfidf_matrix, indices

# ------------------
# Recommendation Logic
# ------------------
def get_recommendations_memory_efficient(drug_name, df, tfidf_matrix, indices):
    """
    Generates drug recommendations based on cosine similarity, calculated on-the-fly.
    """
    if drug_name not in indices:
        return None

    idx = indices[drug_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    drug_vector = tfidf_matrix[idx]
    sim_scores = linear_kernel(drug_vector, tfidf_matrix).flatten()
    similar_drug_indices = sim_scores.argsort()[-6:-1][::-1]
    
    # Fetch more details for a richer display
    rec_details = df.iloc[similar_drug_indices][['drug_name', 'all_uses', 'Therapeutic Class', 'Habit Forming']]
    
    return rec_details

# ------------------
# Streamlit App UI
# ------------------

# --- Sidebar ---
with st.sidebar:
    st.title('💊 PharmaAssist Recommender')
    st.markdown("""
    Welcome to a showcase of the project on **"Converting raw data into valuable insights."**
    
    This app uses a content-based filtering model to help medical professionals discover substitute drugs based on their therapeutic profiles.
    """)
    
    st.markdown("---")
    st.subheader("About the Project")
    st.markdown("""
    - **Model**: TF-IDF Vectorization & Cosine Similarity.
    - **Data**: Based on a dataset of over 200,000 medicines.
    - **Creator**: Abhay Tiwari, a 3rd-year undergraduate at NIT Allahabad, passionate about Data Science and Machine Learning.
    """)
    st.markdown("---")

# --- Main Page ---
st.header('Drug Substitute Recommendation System')
st.markdown("Select a drug from the list below to find the top 5 most similar substitutes.")


# Load data and prepare the model
df, tfidf_matrix, indices = load_data_and_prepare_model()

if df is not None:
    drug_list = df['drug_name'].sort_values().unique()

    # User input selectbox
    selected_drug = st.selectbox(
        'Start typing a drug name...',
        options=drug_list,
        index=None,
        placeholder="Select a drug from the list"
    )

    # Display recommendations when a drug is selected
    if selected_drug:
        with st.spinner('Analyzing similarities and finding the best substitutes...'):
            recommendations = get_recommendations_memory_efficient(selected_drug, df, tfidf_matrix, indices)
            
            st.success(f"Top 5 Recommended Substitutes for **'{selected_drug}'**:")
            
            if recommendations is not None and not recommendations.empty:
                # Create two columns for a card layout
                col1, col2 = st.columns(2)
                
                # Distribute recommendations into the columns
                for i, (index, row) in enumerate(recommendations.iterrows()):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### {i+1}. {row['drug_name']}")
                            st.markdown(f"**🏷️ Therapeutic Class:** `{row['Therapeutic Class']}`")
                            st.markdown(f"**✅ Primary Use(s):** {row['all_uses']}")
                            
                            # Add a visual warning for habit-forming drugs
                            if row['Habit Forming'] == 'YES':
                                st.warning("**⚠️ Habit Forming:** Yes")
                            else:
                                st.info("**Habit Forming:** No")

            else:
                st.error("Could not find suitable recommendations for the selected drug.")
else:
    st.info("Application is ready. Please place the `medicine_data.parquet` file in the app's directory to begin.")

