import pandas as pd
import numpy as np
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

embeddings_file = 'plot_embeddings.npy'


def clean_plot(plot):
    plot = re.sub(r'\W+', ' ', plot.lower())
    words = nltk.word_tokenize(plot)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def filter_movies_by_year(data, min_year=2010):
    return data[data['Release Year'] >= min_year]


def filter_movies_by_country(data, country='American'):
    return data[data['Origin/Ethnicity'] == country]


data = pd.read_csv('wiki_movie_plots_deduped.csv')
data = filter_movies_by_year(data)
data = filter_movies_by_country(data)
data['Cleaned_Plot'] = data['Plot'].apply(clean_plot)

model = SentenceTransformer(
    'paraphrase-MiniLM-L6-v2', cache_folder='model_cache')


def generate_embeddings(text):
    return model.encode(text)


if os.path.exists(embeddings_file):
    # Load embeddings from the file if it exists
    plot_embeddings = np.load(embeddings_file)
else:
    # Calculate embeddings and save them to a file if it doesn't exist
    plot_embeddings = np.vstack(
        data['Cleaned_Plot'].apply(generate_embeddings))
    np.save(embeddings_file, plot_embeddings)


def find_similar_movies_bruteforce(embeddings, query_embedding, k=3):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), embeddings)
    top_k_indices = np.argsort(similarities[0])[-k:][::-1]
    return data.iloc[top_k_indices]


# Select a random movie, print its title and a shortened plot
random_movie = data.sample()
random_title = random_movie['Title'].values[0]
random_plot = random_movie['Plot'].values[0]
random_plot_short = ' '.join(random_plot.split()[:50])

print(f"Random movie: {random_title}")
print(f"Shortened plot: {random_plot_short}...")

# Generate embedding for the random movie's cleaned plot and find similar movies
random_plot_cleaned = random_movie['Cleaned_Plot'].values[0]
random_plot_embedding = generate_embeddings(random_plot_cleaned)
similar_movies = find_similar_movies_bruteforce(
    plot_embeddings, random_plot_embedding, k=5)

print("\nTop 3 similar movies:")
print(similar_movies[['Title', 'Release Year', 'Cleaned_Plot']])
