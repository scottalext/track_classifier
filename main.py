from dotenv import load_dotenv
import kagglehub
import os
import pandas as pd
import requests




def get_historic_data():
    """
    Download Spotify Tracks data set from Kaggle
    """
    kaggle_key = os.getenv("KAGGLE_API_TOKEN")
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset") + "/dataset.csv"
    return path

def generate_data_frame(data):
    """
    Generate pandas data frame for .csv data
    """
    df = pd.read_csv(data)
    return df

def get_live_data():
    session = requests.Session()
    header = {
        "Authorization": "Bearer " + os.getenv("SPOTIFY_CLIENT_SECRET")
    }
    search_url_base = "https://api.spotify.com/v1/search?"
    search_query = "q=artist:ERRA"
    combined_search_url = search_url_base + search_query

    search_response = session.get(combined_search_url, headers=header, timeout=3)
    print(search_response)

def main():
    """
    Main program
    """
    load_dotenv()
    historic_data = get_historic_data()
    historic_df = generate_data_frame(historic_data)
    get_live_data()

if "__main__" == __name__:
    main()