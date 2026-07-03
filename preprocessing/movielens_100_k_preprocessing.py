# -*- coding: utf-8 -*-
"""
MovieLens 100k Dataset Preprocessing Script

This script processes the MovieLens 100k dataset to prepare it for session-based recommender systems.
Steps include:
1. Loading raw data (user interactions and movie metadata).
2. Processing genres and mapping them to movies.
3. Defining user sessions based on a 30-minute time window.
4. Filtering users and movies based on minimum appearance thresholds.
5. Saving the preprocessed dataset as a CSV file.
"""

from google.colab import drive
drive.mount('/content/drive')

# Import required libraries
import pandas as pd
import numpy as np

###############################################
# Step 1: Load raw datasets
###############################################

# Load user interactions
data_fields = ['user_id', 'movie_id', 'rating', 'timestamp']
df_interactions = pd.read_csv(
    r'/content/drive/MyDrive/movilense 100k session/u.data',
    sep='\t',
    names=data_fields,
    encoding='latin-1',
    nrows=1000  # Optional: Load a subset for faster processing
)
print("User interactions loaded:")
print(df_interactions.head(2))

# Load movie metadata
movie_fields = [
    'movie_id', 'title', 'release date', 'video release date', 'IMDb URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
df_movies = pd.read_csv(
    r'/content/drive/MyDrive/movilense 100k session/u.item',
    sep='|',
    names=movie_fields,
    encoding='latin-1'
)
print("Movie metadata loaded:")
print(df_movies.head(5))

###############################################
# Step 2: Process genres
###############################################

# Combine genres into a single column
genres = []
for row in range(len(df_movies)):
    genre_list = [
        column for column in movie_fields[5:] if df_movies[column].iloc[row] == 1
    ]
    genres.append(', '.join(genre_list))
df_movies['genres'] = genres

# Keep only relevant columns
df_movies = df_movies[['movie_id', 'genres']]
print("Processed movie genres:")
print(df_movies.head())

# Filter movies to include only specific genres
relevant_genres = ['Drama', 'Action', 'Comedy', 'Romance', 'Thriller', 'Sci-Fi', 'Adventure']
df_movies = df_movies[df_movies['genres'].str.split(',').apply(
    lambda genre_list: any(genre in relevant_genres for genre in genre_list)
)]

###############################################
# Step 3: Merge interactions with movies
###############################################

# Merge datasets on movie_id
df_merged = pd.merge(df_interactions, df_movies, on='movie_id', how='left')
print("Merged dataset:")
print(df_merged.head())

###############################################
# Step 4: Define sessions based on timestamps
###############################################

def create_sessions(user_df):
    """Assign session IDs based on a 30-minute time window."""
    return user_df.assign(
        session_id=user_df['timestamp']
        .diff()
        .div(pd.Timedelta(minutes=1))
        .gt(30)
        .cumsum()
    )

df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'], unit='s')
df_sessions = df_merged.sort_values(['user_id', 'timestamp']).groupby('user_id').apply(create_sessions).reset_index(drop=True)

###############################################
# Step 5: Filter sessions and users
###############################################

# Filter sessions with more than a threshold of interactions
session_threshold = 50
df_sessions['session_id_modified'] = 0
session_counter = 0
session_offset = 0

for i in range(len(df_sessions) - 1):
    df_sessions.loc[i, 'session_id_modified'] = df_sessions.loc[i, 'session_id'] + session_offset
    if (
        df_sessions.loc[i, 'user_id'] != df_sessions.loc[i + 1, 'user_id']
        or df_sessions.loc[i, 'session_id'] != df_sessions.loc[i + 1, 'session_id']
    ):
        session_counter = 0
    else:
        session_counter += 1
    if session_counter >= session_threshold:
        session_offset += 1
        session_counter = 0

df_sessions.loc[len(df_sessions) - 1, 'session_id_modified'] = (
    df_sessions.loc[len(df_sessions) - 1, 'session_id'] + session_offset
)

# Drop original session ID
filtered_sessions = df_sessions.drop(columns=['session_id'])
filtered_sessions.rename(columns={'session_id_modified': 'session_id'}, inplace=True)

# Filter users with minimum movie appearances
min_movie_appearances = 5
user_appearances = filtered_sessions['user_id'].value_counts()
good_users = user_appearances[user_appearances >= min_movie_appearances].index
filtered_sessions = filtered_sessions[filtered_sessions['user_id'].isin(good_users)]

###############################################
# Step 6: Save preprocessed data
###############################################

output_path = r'/content/drive/MyDrive/movilense 100k session/100k_session.csv'
filtered_sessions.to_csv(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")
