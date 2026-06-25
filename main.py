from collections import Counter
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_historic_data():
    """
    Download Spotify Tracks data set from Kaggle
    """
    kaggle_key = os.getenv("KAGGLE_API_TOKEN")
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset") + "/dataset.csv"
    return path

def load_and_clean_historic_data(data):
    """
    Create data frame and clean data
    """
    df = pd.read_csv(data)

    # drop rows with null values
    print(f"Data set length prior to cleaning: {len(df)}")
    df_clean = df.dropna()
    print(f"Data set length after cleaning: {len(df_clean)}")
    # drop columns with all unique values
    df_clean = df_clean.drop(columns=["track_id", "artists", "album_name", "track_name"])

    # one-hot encoding
    categorical_cols = df_clean.select_dtypes(include=["string", "boolean"]).columns.tolist()
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, dtype=int, drop_first=True)

    # identify which tracks are a 'hit' based on popularity
    hit_idx = df_clean.index[df_clean['popularity'] >= 70]
    print(f"Number of hits from historic data set: {len(hit_idx)}")
    df_clean['hit'] = np.where(df_clean['popularity'] >= 70, 1, 0)
    hit_idx_check = df_clean.index[df_clean['hit'] == 1]
    print(f"verify hit tracks are properly identified: {hit_idx == hit_idx_check}")

    return df_clean

def train_model(clean_df):
    """
    Use Binomial Logistic Regression model to classify hits
    """
    # data oversampling to fix imbalances
    x = clean_df.drop(columns=["hit", "popularity"])
    y = clean_df['hit']
    print(f"Original dataset split: {Counter(y)}")
    sm = SMOTE(random_state=33)
    x_res, y_res = sm.fit_resample(x, y)
    print(f"Resampled dataset split: {Counter(y_res)}")

    # split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=33)

    # create model on training set
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_train, y_train)

    # evaluate model performance
    target_names = ["Flop", "Hit"]
    print("\n\n")
    print(40 * "*")
    print("MODEL PERFORMANCE EVALUATION")
    print(40 * "*")
    y_pred = pipe.predict(x_test)
    print(f"Number of real hits in test set: {len(np.where(y_test == 1)[0])}")
    print(f"Number of hits predicted: {len(np.where(y_pred == 1)[0])}")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

def get_live_data():
    session = requests.Session()

    # Auth
    CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = session.post(auth_url, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    })
    auth_data = auth_response.json()
    access_token = auth_data.get("access_token")

    # Search
    search_url= "https://api.spotify.com/v1/search"
    header = {
        "Authorization": f"Bearer {access_token}"
    }
    query_params = {
        "q": "artist:ERRA",
        "type": "artist",
        "market": "US"
    }

    search_response = session.get(search_url, headers=header, params=query_params, timeout=3)
    if search_response.status_code == 200:
        search_res = search_response.json()
        print(search_res)
    else:
        raise Exception(f"Error: {search_response.status_code}: {search_response.text}")

def main():
    """
    Main program
    """
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv()
    historic_data = get_historic_data()
    cleaned_data = load_and_clean_historic_data(historic_data)
    train_model(cleaned_data)
    # get_live_data()



if "__main__" == __name__:
    main()