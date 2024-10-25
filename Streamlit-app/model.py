import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# Load dataset
dataset = pd.read_csv('SocialMediaUsersDataset.csv')
dataset = dataset.head(10000)

# One-hot encode interests
interests = dataset['Interests'].str.get_dummies(', ')
interests.fillna(0, inplace=True)

# Convert DOB to datetime
dataset['DOB'] = pd.to_datetime(dataset['DOB'])

# Get the current date
current_date = datetime.now()

# Function to calculate age
def calculate_age(dob):
    return current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))

# Apply the function to calculate the 'Age'
dataset['Age'] = dataset['DOB'].apply(calculate_age)

# One-hot encode gender
gender = dataset[['Gender']]
gender_encoded = pd.get_dummies(gender)

# Get age and location data
age = dataset[['Age']]
location = dataset[['City', 'Country']]
location_encoded = pd.get_dummies(location)

# Combine features
features = pd.concat([interests, gender_encoded, age, location_encoded], axis=1)
features = features.astype(int)

# Compute similarity matrix
similarity_matrix = cosine_similarity(features)

# Save features and similarity matrix to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump((features, similarity_matrix), f)

print("Model saved as model.pkl")
