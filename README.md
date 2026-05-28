# Spotify Vibe Classifier & Recommendation System

## Live Demo

https://tracychanty-spotify-vibe.streamlit.app/

## Project Overview

This project is an end-to-end music machine learning system built on Spotify track features. It combines unsupervised learning, supervised classification, and similarity-based recommendation inside a custom Streamlit web app, creating an interactive Spotify-inspired music discovery experience.

The project has two main user flows:
- `Artist Tool`: classify a track into a vibe category using audio features
- `Listener Tool`: discover similar songs based on mood/preferences or a song search

The full workflow is documented in [`code_final.ipynb`](./code_final.ipynb) and deployed through [`web.py`](./web.py).

The pipeline works like this:
1. Clean and preprocess a Spotify track dataset
2. Use K-Means clustering to discover vibe groupings from audio features
3. Name the resulting clusters as interpretable vibe labels
4. Train a Random Forest classifier to predict those vibe labels
5. Build a cosine-similarity recommendation engine for song matching
6. Serve the system in a styled Streamlit interface

## Dataset

Dataset source: Spotify Tracks Dataset from Kaggle 
(https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

- Raw dataset size: `114,000` tracks
- Cleaned modeling dataset: `80,800` tracks
- Genres covered: `114`
- Unique artists: `31,437`
- Unique albums: `46,589`
- Null values after cleaning: `0`

## Machine Learning Components

### Vibe Classification Model

The classification model predicts a song's vibe category using 11 track-level features:
- danceability
- energy
- valence
- tempo
- acousticness
- speechiness
- liveness
- instrumentalness
- loudness
- explicit
- duration_min

The system groups tracks into 4 vibe classes:
- Energetic & Danceable
- Acoustic & Mellow
- Instrumental
- Acoustic & Instrumental

### Recommendation System

The recommendation engine uses a scaled feature matrix and cosine similarity to retrieve similar songs.

It supports two flows:
- Preference-based: users move sliders for mood, energy, danceability, tempo, and related features
- Song-based: users search for a track name and get similar songs

Available filters in the app include:
- vibe filtering
- genre filtering
- result count selection
- same-vibe vs diverse-vibe retrieval

## Model Performance

### Clustering Quality

- K-Means with `k = 4`
- Silhouette score: `0.3165`
- Davies-Bouldin score: `1.1806`

### Classification Performance

The vibe classifier is a `RandomForestClassifier`.

Evaluation results:
- Test accuracy: `0.9905`
- Macro F1-score: `0.9883`
- 5-fold CV macro F1 mean: `0.9871`
- 5-fold CV macro F1 std: `0.0011`

## Web Appplication

The Streamlit app in [`web.py`](./web.py) provides two interfaces:

### Artist Tool

- manual audio-feature input with sliders
- vibe prediction with confidence score
- probability bar chart
- radar chart of vibe probabilities

### Listener Tool

- preference slider recommendations
- song search recommendations
- genre and vibe filters
- styled recommendation cards


## Repository Structure

```text
.
├── code_final.ipynb        # full analysis, preprocessing, clustering, classification, recommendation
├── web.py                  # Streamlit web application
├── style.css               # custom app styling
├── dataset.csv             # original dataset used in the notebook
├── rec_catalogue.csv       # exported recommendation catalogue
├── vibe_classifier.pkl     # trained Random Forest classifier
├── vibe_names.pkl          # saved vibe label mapping
├── rec_scaler.pkl          # fitted scaler for recommendation features
├── rec_matrix.npy          # scaled matrix used for cosine similarity
├── spotify_logo.png        # app branding asset
├── requirements.txt 
└── README.md
```

## Technologies Used

### Programming & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib

### Visualization
- Plotly
- Matplotlib
- Seaborn

### Web Development
- Streamlit
- HTML/CSS

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tracychanty/spotify-classification-recommendation-system.git
```

### 2. Navigate to the project folder

```bash
cd spotify-classification-recommendation-system
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the web app

```bash
streamlit run web.py
```
