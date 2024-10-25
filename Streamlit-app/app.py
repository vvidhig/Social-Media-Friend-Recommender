import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed features and similarity matrix
@st.cache_resource
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            features, similarity_matrix = pickle.load(f)
        return features, similarity_matrix
    except EOFError:
        st.error("Failed to load the recommendation model. Please ensure 'social_media_recommendation.pkl' is in the correct directory and is not corrupted.")
        return None, None

features, similarity_matrix = load_model()

if features is None or similarity_matrix is None:
    st.stop()  # Stops further execution if model fails to load


# Load dataset for display purposes
@st.cache_data
def load_data():
    return pd.read_csv('SocialMediaUsersDataset.csv').head(10000)

dataset = load_data()

# Streamlit app UI
st.title("Social Media Friend Recommendation App")

# User inputs
st.header("Enter Your Details")

interests_input = st.text_area("Enter your interests, separated by commas (e.g., 'Movies, Fashion, Books')")
gender_input = st.selectbox("Select your gender", ["Male", "Female"])
age_input = st.number_input("Enter your age", min_value=10, max_value=100)
city_input = st.selectbox("Select your city", dataset['City'].unique())
country_input = st.selectbox("Select your country", dataset['Country'].unique())

if st.button("Get Friend Recommendations"):
    # Convert user inputs to a feature vector matching the model's feature format
    def create_user_features(interests, gender, age, city, country):
        # One-hot encode interests
        interests_vector = features.columns.str.contains('|'.join(interests.split(', ')), case=False).astype(int)

        # One-hot encode gender
        gender_vector = pd.get_dummies(pd.Series([gender]), drop_first=False).reindex(columns=['Gender_Female', 'Gender_Male']).fillna(0).values.flatten()

        # Age
        age_vector = np.array([age])

        # One-hot encode location
        location_vector = pd.get_dummies(pd.DataFrame({'City': [city], 'Country': [country]}), drop_first=False).reindex(columns=features.columns[features.columns.str.contains("City_|Country_")]).fillna(0).values.flatten()

        # Combine all parts of the feature vector
        return np.concatenate([interests_vector, gender_vector, age_vector, location_vector])

    user_features = create_user_features(interests_input, gender_input, age_input, city_input, country_input).reshape(1, -1)

    # Calculate similarity with all users in the dataset
    similarities = cosine_similarity(user_features, features).flatten()

    # Get top 5 most similar users (excluding the user themself)
    top_indices = similarities.argsort()[-6:-1][::-1]
    recommendations = dataset.iloc[top_indices][['UserID', 'Name', 'Interests', 'City', 'Country']]

    st.write("Top 5 Friend Recommendations:")
    st.table(recommendations)