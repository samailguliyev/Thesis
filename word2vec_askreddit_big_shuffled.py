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
from collections import Counter
from scipy.stats import spearmanr
import pandas as pd
from gensim.models import Word2Vec
import random
import pandas as pd
from datetime import datetime
import re



# Initialize logging
logging.basicConfig(filename='Big_Askreddit_model_training_random.log', level=logging.INFO)


keywords = ['tinder', 'zoom', 'telegram', 'twitch', 'discord',
             'teams', 'slack', 'stripe', 'snap', 'swift',
             'bumble', 'medium', 'hinge', 'notion', 'signal', 
             'azure', 'echo', 'alexa', 'prime', 'edge',
             'meta', 'lightning', 'eats', 'spark', 'hana',
             'ring', 'square', 'corona', 'vegan', 'vegans', 'veganism']

# Generate special tokens
special_tokens = []
years = range(2009, 2023)
for keyword in keywords:
    for year in years:
        special_tokens.append(f"{keyword}_{year}")


chunk_size = 4000000  # You can adjust this size based on your available memory


# Create a regular expression pattern to match any of the keywords
keyword_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b')

# Initialize an empty list to hold the modified sentences
word2vec_data = []

# Reading the CSV file in chunks
for i, chunk in enumerate(pd.read_csv(
    "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe_reshuffled_random.csv", sep='\t', chunksize=chunk_size)):
    print(i)

    # Drop NA values
    chunk = chunk.dropna(subset=['preprocessed_body', 'created_utc', 'reshuffled_year'])
    # Use reshuffled_year and ensure it's a string type
    chunk['year'] = chunk['reshuffled_year'].astype(str)


    # Vectorized check for sentences containing any of the keywords
    contains_keyword = chunk['preprocessed_body'].str.contains(keyword_pattern)

    # Function to replace keywords
    def replace_keywords(sentence, year):
        words = sentence.split()
        return ' '.join([f"{word}_{year}" if word in keywords else word for word in words])

    # Replace tokens only in filtered ones
    chunk.loc[contains_keyword, 'modified_body'] = chunk.loc[contains_keyword].apply(lambda row: replace_keywords(row['preprocessed_body'], row['year']), axis=1)
    # For rows without keywords, simply use the original text
    chunk.loc[~contains_keyword, 'modified_body'] = chunk.loc[~contains_keyword, 'preprocessed_body']

    # Convert the 'modified_body' column to a list of lists (each inner list is a sentence)
    modified_sentences = chunk['modified_body'].str.split().tolist()

    # Extend the word2vec_data list
    word2vec_data.extend(modified_sentences)





print("here")


# Delete the DataFrame to free up memory
print("here")

# Read SimLex dataset
simlex_data = pd.read_csv('/home/ma/ma_ma/ma_sguliyev/New_thesis_repository/Data/SimLex-999/SimLex-999.txt', sep='\t')

# Evaluation function
def evaluate_model(model):
    simlex_pairs = []
    model_pairs = []
    for i, row in simlex_data.iterrows():
        word1, word2, score = row['word1'], row['word2'], row['SimLex999']
        if word1 in model.wv and word2 in model.wv:
            simlex_pairs.append(score)
            model_pairs.append(model.wv.similarity(word1, word2))
    return spearmanr(simlex_pairs, model_pairs)





# Best Params for 100% data
best_params = {
    'vector_size': 300,
    'window': 2,
    'min_count': 300,
    'sample': 0.005,
    'negative': 30,
    'alpha': 0.025,
    'sg': 1,
    'workers': 512,
    'batch_words': 100,
    'hs': 0
}

# List of sampling percentages
sampling_percentages = [1, 100, 50, 25, 10, 5]

for percentage in sampling_percentages:
    print(f"Training model with {percentage}% of data")

    # Sample data based on the given percentage
    if percentage < 100:
        sample_size = int((percentage / 100) * len(word2vec_data))
        sample_indices = random.sample(range(len(word2vec_data)), k=sample_size)
        sampled_data = [word2vec_data[i] for i in sample_indices]
    else:
        sampled_data = word2vec_data

    # Adjust min_count based on the sample size
    adjusted_min_count = int(300 * (percentage / 100))
    best_params['min_count'] = max(1, adjusted_min_count)  # Ensuring min_count is at least 1

    start_time = datetime.now()
    model = Word2Vec(**best_params)

    print("Building vocabulary...")
    vocab_start_time = time.time()

    vocab = list(model.wv.index_to_key)
    word_freq = Counter()
    for sentence in sampled_data:
        word_freq.update(sentence)

    # Update the frequency of special tokens to a high value
    for token in special_tokens:
        word_freq[token] = 99999  # set to some high number

    model.build_vocab_from_freq(word_freq)

    model.train(sampled_data, total_examples=len(sampled_data), epochs=1)
    end_time = datetime.now()

    correlation, p_value = evaluate_model(model)

    log_message = f"Random Sampling Percentage: {percentage}%, Time: {start_time} to {end_time}, Correlation: {correlation}, p-value: {p_value}"
    logging.info(log_message)

    model.save(f"random_sampled_word2vec_model_{percentage}_percent.model")

print("Training completed for all sampling percentages.")
