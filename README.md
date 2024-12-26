# Social Media Friend Recommendation System

![Demo](demo.gif)

## Features

- **User Input**: Users can input their gender, age, city, country, and interests.
- **Filters**: Users can set filters for maximum age difference, matching from the same city, and minimum common interests.
- **Recommendations**: The app provides top 5 friend recommendations based on cosine similarity of user features.
- **Interactive UI**: Built with Streamlit for an interactive and user-friendly interface.


**Explanation:**
- **`requirements.txt`**: Contains the dependencies required to run the project.
- **`SocialMediaUsersDataset.csv`**: The dataset used for recommendations.
- **`SocialMediaUsersRecommendation.ipynb`**: Jupyter notebook for data analysis and model development.
- **`Streamlit-app/`**: Directory for the Streamlit application.
  - **`app.py`**: Streamlit application code.
  - **`model.pkl`**: Serialized machine learning model.
  - **`model.py`**: Script to create and save the ML model.
  - **`requirements.txt`**: Dependencies for the Streamlit app.
  - **`SocialMediaUsersDataset.csv`**: Dataset used in the app.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/social-media-friend-finder.git
    cd social-media-friend-finder
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Navigate to the [Streamlit-app](http://_vscodecontentref_/9) directory and install its dependencies:
    ```sh
    cd Streamlit-app
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the Dataset**:
    - Ensure [SocialMediaUsersDataset.csv](http://_vscodecontentref_/10) is in the root directory and [Streamlit-app](http://_vscodecontentref_/11) directory.

2. **Train the Model**:
    - Run the [model.py](http://_vscodecontentref_/12) script to preprocess the data and save the model:
        ```sh
        python model.py
        ```

3. **Run the Application**:
    - Start the Streamlit app:
        ```sh
        streamlit run app.py
        ```

4. **Interact with the App**:
    - Open the provided URL in your browser and start finding friends based on your interests and preferences.

## Notebooks

- **SocialMediaUsersRecommendation.ipynb**: Jupyter notebook for data exploration and feature extraction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

