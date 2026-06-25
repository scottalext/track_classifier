# **track_classifier**

This project is a simple track classifier using historical data sets and data grabbed from Spotify's web API.
A simple binomial logistic regression model is trained on historic Spotify data to determine if a song is
a 'hit' or a 'flop'. You will need to have Kagglehub and Spotify api keys to run this on your machine.

## **Setup**

Clone the repository to your machine.

Install the required dependencies via pip. (It is recommended to do this in a virtual env).
```
pip install -r requirements.txt
```

Setup a .env file with the following variables:
```
KAGGLE_API_TOKEN=<your_kaggle_api_token>
SPOTIFY_CLIENT_ID=<your_spotify_api_client_id>
SPOTIFY_CLIENT_SECRET=<your_spotify_api_client_secret>
```

## **Run**

Execute the main.py file via terminal
```
python3 main.py
```