from collections import Counter
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
import kagglehub
import librosa
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
    # drop unnecessary columns
    df_clean = df_clean.drop(columns=["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "time_signature", "track_genre", "explicit"])

    # one-hot encoding
    categorical_cols = df_clean.select_dtypes(include=["string", "boolean"]).columns.tolist()
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype("category")
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, dtype=int, drop_first=True)

    # identify which tracks are a "hit" based on popularity
    hit_idx = df_clean.index[df_clean["popularity"] >= 70]
    print(f"Number of hits from historic data set: {len(hit_idx)}")
    df_clean["hit"] = np.where(df_clean["popularity"] >= 70, 1, 0)
    hit_idx_check = df_clean.index[df_clean["hit"] == 1]
    print(f"verify hit tracks are properly identified: {hit_idx == hit_idx_check}")

    return df_clean

def train_model(clean_df, show_conf_matrix=False):
    """
    Use Binomial Logistic Regression model to classify hits
    """
    # data oversampling to fix imbalances
    x = clean_df.drop(columns=["hit", "popularity"])
    y = clean_df["hit"]
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

    if show_conf_matrix:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()
    
    return pipe

def generate_audio_features(audio_file, audio_path=".songs/"):
    """
    Function to generate audio features from .mp3 file
    """
    if not audio_file.endswith(".mp3"):
        audio_file = audio_file + ".mp3"
    full_path = os.path.join(audio_path, audio_file)
    if os.path.isfile(full_path):
        print("Song found!")
    else:
        raise Exception(f"File not found: {full_path}")
    
    # approximation of deprecated Spotify audio-features endpoint
    audio_features = {
        "duration_ms": None,
        "danceability": None,
        "energy": None,
        "key": None,
        "loudness": None,
        "mode": None,
        "speechiness": None,
        "acousticness": None,
        "instrumentalness": None,
        "liveness": None,
        "valence": None,
        "tempo": None
    }
    y, sr = librosa.load(full_path, sr=22050) # down-sampled for faster processing
    duration = librosa.get_duration(y=y, sr=sr)
    audio_features["duration_ms"] = [int(duration * 1000)]
    
    # loudness approximation
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    loudness = librosa.amplitude_to_db(np.array([mean_rms]), ref=1.0)[0]
    audio_features["loudness"] = [round(float(loudness), 3)]

    # danceability approximation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_pulses = onset_env[beats] if len(beats) > 0 else [0]
    danceability = float(np.clip(np.mean(beat_pulses) / 10.0, 0.0, 1.0))
    audio_features["tempo"] = [round(float(tempo[0] if isinstance(tempo, np.ndarray) else tempo), 1)]
    audio_features["danceability"] = [round(danceability, 3)]

    # energy, acousticness, and speechiness approximation
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    normalized_centroid = np.mean(spec_centroid) / (sr / 2.0)
    energy = float(np.clip(mean_rms * 5 + normalized_centroid * 2, 0.0, 1.0))
    flatness = librosa.feature.spectral_flatness(y=y)
    acousticness = float(np.clip(1.0 - (energy * 0.7 + np.mean(flatness) * 3), 0.0, 0.1))
    liveness = float(np.clip(np.std(rms) * 4, 0.0, 1.0))
    instrumentalness = float(1.0 - np.mean(flatness))
    audio_features["energy"] = [round(energy, 3)]
    audio_features["acousticness"] = [round(acousticness, 3)]
    audio_features["speechiness"] = [round(float(np.mean(flatness)), 3)]
    audio_features["liveness"] = [round(liveness, 3)]
    audio_features["instrumentalness"] = [round(instrumentalness, 3)]

    # key and mode detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)
    # major and minor chord profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    major_corr = [np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1] for i in range(12)]
    minor_corr = [np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1] for i in range(12)]

    if max(major_corr) > max(minor_corr):
        key = int(np.argmax(major_corr))
        mode = 1
    else:
        key = int(np.argmax(minor_corr))
        mode = 0
    audio_features["key"] = [key]
    audio_features["mode"] = [mode]

    # valence approximation
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    mode_score = np.mean(tonnetz[0])
    normalized_tempo = np.clip((tempo - 60) / 120, 0.0, 1.0)
    raw_valence = (0.4 * mode_score) + (0.3 * normalized_centroid) + (0.3 * normalized_tempo)
    valence = 1 / (1 + np.exp(-raw_valence * 5))
    audio_features["valence"] = [round(valence[0], 3)]

    track_df = pd.DataFrame(audio_features)
    
    print(f"Approximation of Spotify audio features:\n{track_df}")
    return track_df

def main():
    """
    Main program
    """
    os.system("cls" if os.name == "nt" else "clear")
    print("Welcome to track_classifier!\n\n")
    load_dotenv()
    historic_data = get_historic_data()
    cleaned_data = load_and_clean_historic_data(historic_data)
    model = train_model(cleaned_data)
    song_name = input("Provide audio file name: ")
    track_features = generate_audio_features(song_name)
    track_prediction = model.predict(track_features)[0]
    if track_prediction == 1:
        print("According to the model, this song is a hit!")
    else:
        print("According to the model, this song is a flop!")



if "__main__" == __name__:
    main()