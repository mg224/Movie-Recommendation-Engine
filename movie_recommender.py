import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# Read CSV file with Pandas
df = pd.read_csv("movie_dataset.csv")
# Test: 
# print(df.head())
# print(df.columns)

# Select Features. For this we will recommend movies based on cast, genre, and director
features = ['keywords', 'cast', 'genres', 'director']

# Create a column in DF that combines all selected features in one string
for feature in features:
    df[feature] = df[feature].fillna('')
    
def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error:", row)

df["combined_features"] = df.apply(combine_features, axis=1)

#Test: 
# print("Combined Features:", df["combined_features"].head())

# Create a matrix count_matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# Compute cosine similarity based on count_matrix
cosine_sim = cosine_similarity(count_matrix)

# Get user's favorite movie
user_favorite_movie = input("What is your favorite movie? ")

# Get index of user's favorite movie from the title
movie_index = get_index_from_title(user_favorite_movie)

similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

# Get a list of similar movies in descending order of similarity score
print("Recommended Movies: ")
i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i > 50:
        break