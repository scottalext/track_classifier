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
    load_dotenv()
    historic_data = get_historic_data()
    historic_df = generate_data_frame(historic_data)
    get_live_data()

if "__main__" == __name__:
    main()