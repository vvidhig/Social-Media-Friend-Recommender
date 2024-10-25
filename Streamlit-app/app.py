import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Social Media Friend Finder",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:


@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('SocialMediaUsersDataset.csv')
        return df.head(10000)  # Match the model size
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
features, similarity_matrix, dataset = load_model()

if dataset is None or features is None or similarity_matrix is None:
    st.error("Failed to load necessary data. Please check your data files.")
    st.stop()

def create_user_features(interests, gender, age, city, country, features_df):
    """Create feature vector matching the model's feature structure."""
    try:
        # Initialize a zero vector with the same columns as the training features
        user_features = pd.DataFrame(0, index=[0], columns=features_df.columns)
        
        # Set interest features
        for interest in interests:
            matching_cols = [col for col in features_df.columns if interest.lower() in col.lower()]
            for col in matching_cols:
                user_features[col] = 1
        
        # Set gender feature
        gender_col = f"Gender_{gender}"
        if gender_col in user_features.columns:
            user_features[gender_col] = 1
        
        # Set city feature
        city_col = f"City_{city}"
        if city_col in user_features.columns:
            user_features[city_col] = 1
        
        # Set country feature
        country_col = f"Country_{country}"
        if country_col in user_features.columns:
            user_features[country_col] = 1
            
        return user_features
        
    except Exception as e:
        st.error(f"Error creating user features: {str(e)}")
        return None

def get_recommendations(user_features, features_df, similarity_matrix, dataset_df, 
                       user_age, max_age_diff, same_city, min_common_interests):
    """Get recommendations with age-based filtering."""
    try:
        # Calculate similarities
        user_similarities = cosine_similarity(user_features, features_df)[0]
        
        # Create recommendations dataframe
        recommendations = dataset_df.copy()
        recommendations['similarity'] = user_similarities
        
        # Filter by age
        recommendations = recommendations[
            (recommendations['Age'] >= user_age - max_age_diff) & 
            (recommendations['Age'] <= user_age + max_age_diff)
        ]
        
        # Filter by city if requested
        if same_city:
            recommendations = recommendations[recommendations['City'] == city_input]
        
        # Filter by minimum common interests
        recommendations['common_interests'] = recommendations['Interests'].apply(
            lambda x: len(set(x.split(', ')) & set(selected_interests))
        )
        recommendations = recommendations[
            recommendations['common_interests'] >= min_common_interests
        ]
        
        # Get top 5 recommendations
        top_recommendations = recommendations.nlargest(5, 'similarity')
        
        return top_recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def display_recommendations(recommendations_df, user_interests):
    if recommendations_df is None or recommendations_df.empty:
        st.warning("No matches found with current filters. Try adjusting your preferences!")
        return
        
    st.header("🎉 Your Top Matches")
    
    for _, row in recommendations_df.iterrows():
        with st.container():
            recommendation_interests = set(row['Interests'].split(', '))
            common_interests = set(user_interests) & recommendation_interests
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>{row['Name']}</h3>
                <p>📍 {row['City']}, {row['Country']}</p>
                <p>👤 {row['Gender']} • 🎂 {row['Age']} years old</p>
                <p>🌟 Similarity Score: {row['similarity']:.2%}</p>
                <p>💫 Common Interests: {', '.join(common_interests)}</p>
            </div>
            """, unsafe_allow_html=True)

# Main app
st.title("👥 Social Media Friend Finder")
st.write("Find like-minded friends based on your interests and preferences!")

# Sidebar filters
with st.sidebar:
    st.header("Filter Settings")
    max_age_diff = st.slider("Maximum Age Difference", 0, 50, 10)
    same_city = st.checkbox("Match from same city only", False)
    min_common_interests = st.slider("Minimum Common Interests", 0, 10, 2)

# User inputs
col1, col2 = st.columns(2)
with col1:
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    age_input = st.number_input("Age", min_value=13, max_value=100, value=25)
with col2:
    city_input = st.selectbox("City", sorted(dataset['City'].unique()))
    country_input = st.selectbox("Country", sorted(dataset['Country'].unique()))

# Interest selection
st.header("🌟 Select Your Interests")

# Get unique interests from dataset
all_interests = set()
for interests_str in dataset['Interests'].str.split(', '):
    all_interests.update(interests_str)
interest_list = sorted(list(all_interests))

# Search functionality
search_term = st.text_input("Search interests", "")
filtered_interests = [
    interest for interest in interest_list 
    if search_term.lower() in interest.lower()
] if search_term else interest_list

# Display interests in columns
cols = st.columns(4)
selected_interests = []
for idx, interest in enumerate(filtered_interests):
    with cols[idx % 4]:
        if st.checkbox(interest, key=f"interest_{interest}"):
            selected_interests.append(interest)

# Get recommendations
if st.button("🔍 Find Friends", use_container_width=True):
    if not selected_interests:
        st.warning("Please select at least one interest!")
    else:
        with st.spinner("Finding your perfect matches..."):
            user_features = create_user_features(
                selected_interests, gender_input, age_input,
                city_input, country_input, features
            )
            
            if user_features is not None:
                recommendations = get_recommendations(
                    user_features, features, similarity_matrix, dataset,
                    age_input, max_age_diff, same_city, min_common_interests
                )
                
                if recommendations is not None:
                    display_recommendations(recommendations, selected_interests)