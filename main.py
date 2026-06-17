from dotenv import load_dotenv
import kagglehub
import os
import pandas as pd

load_dotenv()

# Historical data
kaggle_key = os.getenv("KAGGLE_API_TOKEN")

path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset") + "/dataset.csv"

df_hist = pd.read_csv(path)
print(df_hist.head())