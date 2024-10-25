import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import Tuple

# Constants
DATASET_FILE = 'SocialMediaUsersDataset.csv'
MAX_DATASET_SIZE = 10000

def load_dataset(file_path: str, max_size: int) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        file_path (str): Path to dataset file.
        max_size (int): Maximum number of rows.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        dataset = pd.read_csv(file_path)
        return dataset.head(max_size)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


def calculate_age(dob: pd.Timestamp) -> int:
    """
    Calculate age from date of birth.

    Args:
        dob (pd.Timestamp): Date of birth.

    Returns:
        int: Calculated age.
    """
    current_date = datetime.now()
    return current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))


def preprocess_data(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess dataset.

    Args:
        dataset (pd.DataFrame): Raw dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed features and similarity matrix.
    """
    # One-hot encode interests
    interests = dataset['Interests'].str.get_dummies(', ')
    interests.fillna(0, inplace=True)

    # Convert DOB to datetime and calculate age
    dataset['DOB'] = pd.to_datetime(dataset['DOB'])
    dataset['Age'] = dataset['DOB'].apply(calculate_age)

    # One-hot encode gender
    gender_encoded = pd.get_dummies(dataset['Gender'], prefix="Gender")

    # One-hot encode location (city and country separately to avoid mismatches)
    city_encoded = pd.get_dummies(dataset['City'], prefix="City")
    country_encoded = pd.get_dummies(dataset['Country'], prefix="Country")

    # Concatenate all feature sets
    features = pd.concat([interests, gender_encoded, dataset[['Age']], city_encoded, country_encoded], axis=1)
    features = features.astype(int)

    assert "Age" in features.columns.tolist(), "Age column missing"

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(features.values)

    return features, similarity_matrix


def save_model(features: pd.DataFrame, similarity_matrix: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """
    Save preprocessed features, similarity matrix, and dataset to a pickle file.

    Args:
        features (pd.DataFrame): Preprocessed features.
        similarity_matrix (pd.DataFrame): Similarity matrix.
        dataset (pd.DataFrame): Original dataset.
    """
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump((features, similarity_matrix, dataset), f)
        print("Model saved as model.pkl")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


if __name__ == "__main__":
    dataset = load_dataset(DATASET_FILE, MAX_DATASET_SIZE)
    if dataset is not None:
        features, similarity_matrix = preprocess_data(dataset)
        save_model(features, similarity_matrix, dataset)