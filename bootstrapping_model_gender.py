from gensim.models import Word2Vec
import pandas as pd
import logging
from scipy.stats import spearmanr
from datetime import datetime
import math
import random
from collections import Counter
import os
import json
import time
import re

# Initialize logging
logging.basicConfig(filename='Gender_LOG_bootstrapped_knownfinal_model_training.log', level=logging.INFO)

keywords = ['tinder', 'zoom', 'telegram', 'twitch', 'discord', 'teams', 'slack', 'stripe', 'snap', 'swift',
             'bumble', 'medium', 'hinge', 'notion', 'signal', 'azure', 
             'echo', 'alexa', 'prime', 'edge', 'meta', 'lightning', 'eats', 
             'spark', 'hana', 'ring', 'square', 'corona', 'vegan', 'vegans', 'veganism']

subreddits = ['askwomen', 'askmen']

# Generate special tokens
special_tokens = []
years = range(2013, 2023)
for subreddit in subreddits:
    for keyword in keywords:
        for year in years:
            special_tokens.append(f"{subreddit}_{keyword}_{year}")

chunk_size = 2000000

def process_dataframe(df_path, subreddit):
    word2vec_data = []
    keyword_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b')

    for i, chunk in enumerate(pd.read_csv(df_path, sep='\t', chunksize=chunk_size)):
        print(f"Processing chunk {i} for {subreddit}... Total rows: {len(chunk)}")

        chunk = chunk.dropna(subset=['preprocessed_body', 'created_utc'])
        
        # Convert 'year' column to string to match the filtering criteria
        chunk['year'] = chunk['year'].astype(str)

        # Filter data and print diagnostic info
        chunk = chunk[chunk['year'].isin([str(y) for y in years])]
        print(f"Rows matching the years filter: {len(chunk)}")

        contains_keyword = chunk['preprocessed_body'].str.contains(keyword_pattern)

        def replace_keywords(sentence, year):
            words = sentence.split()
            return ' '.join([f"{subreddit}_{word}_{year}" if word in keywords else word for word in words])

        modified_bodies = chunk[contains_keyword].apply(lambda row: replace_keywords(row['preprocessed_body'], row['year']), axis=1)
        
        # Preallocate 'modified_body' column if it doesn't exist yet
        if 'modified_body' not in chunk.columns:
            chunk['modified_body'] = chunk['preprocessed_body']

        # Existing code continues...
        chunk.loc[contains_keyword, 'modified_body'] = chunk.loc[contains_keyword].apply(lambda row: replace_keywords(row['preprocessed_body'], row['year']), axis=1)
        chunk.loc[~contains_keyword, 'modified_body'] = chunk.loc[~contains_keyword, 'preprocessed_body']

        
        modified_sentences = chunk['modified_body'].str.split().tolist()
        word2vec_data.extend(modified_sentences)

    return word2vec_data

# Process both dataframes and combine into one
word2vec_data = process_dataframe(
    "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskWomen_download/AskWomen_per_year_comments_splitted/original_dataframe.csv",
     "askwomen"
    ) + process_dataframe(
        "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskMen_download/AskMen_per_year_comments_splitted/original_dataframe.csv",
         "askmen")

# Rest of the code (unchanged)
print("Starting bootstrapping...")
print(f"{len(word2vec_data)}")

simlex_data = pd.read_csv('/home/ma/ma_ma/ma_sguliyev/New_thesis_repository/Data/SimLex-999/SimLex-999.txt', sep='\t')

def evaluate_model(model):
    simlex_pairs = []
    model_pairs = []
    for i, row in simlex_data.iterrows():
        word1, word2, score = row['word1'], row['word2'], row['SimLex999']
        if word1 in model.wv and word2 in model.wv:
            simlex_pairs.append(score)
            model_pairs.append(model.wv.similarity(word1, word2))
    return spearmanr(simlex_pairs, model_pairs)

best_params = {
    'vector_size': 300,
    'window': 2,
    'min_count': 'TODO',
    'sample': 0.005,
    'negative': 30,
    'alpha': 0.025,
    'sg': 1,
    'workers': 512,
    'batch_words': 100,
    'hs': 0
}

for i in range(1, 10):
    print(f"Training model {i}...")

    sample_indices = random.choices(range(len(word2vec_data)), k=int(len(word2vec_data) * 0.5))
    bootstrap_data = [word2vec_data[i] for i in sample_indices]
    min_count = math.ceil(25*3)
    best_params['min_count'] = min_count

    start_time = datetime.now()
    model = Word2Vec(**best_params)

    print("Building vocabulary...")
    vocab_start_time = time.time()

    vocab = list(model.wv.index_to_key)
    word_freq = Counter()
    for sentence in bootstrap_data:
        word_freq.update(sentence)
    for token in special_tokens:
        word_freq[token] = 99999

    model.build_vocab_from_freq(word_freq)
    model.train(bootstrap_data, total_examples=len(bootstrap_data), epochs=1)
    end_time = datetime.now()

    correlation, p_value = evaluate_model(model)
    log_message = f"Bootstrapping Iteration: {i}, Time: {start_time} to {end_time}, Correlation: {correlation}, p-value: {p_value}"
    logging.info(log_message)

    model.save(f"gendered_models/gendered_newest_bootstrap_word2vec_model_{9+i}.model")

print("Completed bootstrapping.")
