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

# One-hot encode categorical features like gender, city, and country
gender_encoded = pd.get_dummies(dataset['Gender'], prefix="Gender")
location_encoded = pd.get_dummies(dataset[['City', 'Country']], prefix=['City', 'Country'])

# Concatenate all features
features = pd.concat([interests, gender_encoded, dataset[['Age']], location_encoded], axis=1)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(features.values)

# Save the features and similarity matrix to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump((features, similarity_matrix), f)

print("Model saved as model.pkl")
