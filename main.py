from dotenv import load_dotenv
import kagglehub
import os
import pandas as pd




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

def main():
    load_dotenv()
    historic_data = get_historic_data()
    historic_df = generate_data_frame(historic_data)
    print(historic_df.head())

if "__main__" == __name__:
    main()