import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load dataset
dataset = pd.read_csv('SocialMediaUsersDataset.csv')
dataset = dataset.head(10000)

# One-hot encode interests
interests = dataset['Interests'].str.get_dummies(', ')
interests.fillna(0, inplace=True)

# Convert DOB to datetime and calculate age
dataset['DOB'] = pd.to_datetime(dataset['DOB'])
current_date = datetime.now()

def calculate_age(dob):
    return current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))

dataset['Age'] = dataset['DOB'].apply(calculate_age)

# One-hot encode gender
gender_encoded = pd.get_dummies(dataset['Gender'], prefix="Gender")

# One-hot encode location (city and country separately to avoid mismatches)
city_encoded = pd.get_dummies(dataset['City'], prefix="City")
country_encoded = pd.get_dummies(dataset['Country'], prefix="Country")

# Combine features
features = pd.concat([interests, gender_encoded, age, location_encoded], axis=1)
features = features.astype(int)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(features.values)

# Save features and similarity matrix to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump((features, similarity_matrix), f)

print("Model saved as model.pkl")
