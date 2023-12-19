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

# Initialize logging
logging.basicConfig(filename='LOG_bootstrapped_knownfinal_model_training.log', level=logging.INFO)

# # Initialize an empty list to hold the tokenized sentences


# # Read the DataFrame in chunks
# chunk_size = 1_000_000  # You can adjust this size based on your available memory
# for i, chunk in enumerate(pd.read_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe.csv", sep='\t', chunksize=chunk_size)):
#     print(i)
#     word2vec_data.extend([line.split() for line in chunk['preprocessed_body'].dropna().tolist()])

# Rest of your code



import pandas as pd
from datetime import datetime
import re


keywords = ['tinder', 'zoom', 'telegram', 'twitch', 'discord', 'teams', 'slack', 'stripe', 'snap', 'swift',
             'bumble', 'medium', 'hinge', 'notion', 'signal', 'azure', 
             'echo', 'alexa', 'prime', 'edge', 'meta', 'lightning', 'eats', 
             'spark', 'hana', 'ring', 'square', 'corona', 'vegan', 'vegans', 'veganism',
             'askmenvegan', 'askmenvegans', 'askmenveganism','askwomenveganism', 'askwomenvegans', 'askwomenvegan']

# Generate special tokens
special_tokens = []
years = range(2009, 2023)
for keyword in keywords:
    for year in years:
        special_tokens.append(f"{keyword}_{year}")


chunk_size = 4000000  # You can adjust this size based on your available memory
word2vec_data = []
for i, chunk in enumerate(pd.read_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/df_vegan_related.csv", sep='\t', chunksize=chunk_size)):

    print(i)
    chunk_list = chunk['preprocessed_body'].dropna().tolist()
    created_utc_list = chunk['created_utc'].dropna().tolist()

    for idx, line in enumerate(chunk_list):
        sentence = line.split()
        year = datetime.strptime(created_utc_list[idx], '%Y-%m-%d %H:%M:%S').year  # Extract year from the 'created_utc' column

        for j, word in enumerate(sentence):
            if word in keywords:
                sentence[j] = f"{word}_{year}"

        word2vec_data.extend([sentence])






# Create a regular expression pattern to match any of the keywords
keyword_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b')

# Initialize an empty list to hold the modified sentences
word2vec_data = []

# Reading the CSV file in chunks
for i, chunk in enumerate(pd.read_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe.csv", sep='\t', chunksize=chunk_size)):
    print(i)

    # Drop NA values
    chunk = chunk.dropna(subset=['preprocessed_body', 'created_utc'])
    # Extract year from the 'created_utc' column
    chunk['year'] = pd.to_datetime(chunk['created_utc']).dt.year.astype(str)

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

# # Sample sizes
# sample_sizes = [100, 50, 25, 10, 5, 1, 0.5]

# for sample_size in sample_sizes:
#     sample_indices = random.sample(range(len(word2vec_data)), int(len(word2vec_data) * sample_size / 100))
#     sampled_data = [word2vec_data[i] for i in sample_indices]
#     print("here")


#     min_count = math.ceil(sample_size * 3)

#     best_params = {
#         'vector_size': 300,
#         'window': 2,
#         'min_count': min_count,
#         'sample': 0.005,
#         'negative': 20,
#         'alpha': 0.025,
#         'sg': 1,
#         'workers': 64,
#         'batch_words': 100,
#         'hs': 0
#     }


#     # Initialize model and build vocabulary once
#     start_time = datetime.now()
#     model = Word2Vec(**best_params)
#     model.build_vocab(sampled_data)

#     best_correlation = None
#     best_model = None

#     # Train the model
#     for epoch in range(1, 4):
#         model.train(sampled_data, total_examples=model.corpus_count, epochs=1)

#         end_time = datetime.now()
#         correlation, p_value = evaluate_model(model)

#         log_message = f"Sample Size: {sample_size}%, Epoch: {epoch}, Time: {start_time} to {end_time}, Correlation: {correlation}, p-value: {p_value}"
#         logging.info(log_message)

#         if best_correlation is None or correlation > best_correlation:
#             best_correlation = correlation
#             best_model = model

#     # Save the model
#     best_model.save(f"best_word2vec_model_{sample_size}percent.model")



# Bootstrapping
for i in range(1,10):
    print(f"model {i}")

    sample_indices = random.choices(range(len(word2vec_data)), k=int(len(word2vec_data) * 0.5))
    bootstrap_data = [word2vec_data[i] for i in sample_indices]

    min_count = math.ceil(50 * 3)

    best_params['min_count'] = min_count

    start_time = datetime.now()
    model = Word2Vec(**best_params)


    print("Building vocabulary...")
    vocab_start_time = time.time()

    # Assuming word2vec_data is already loaded
        # model.build_vocab(word2vec_data, progress_per=1000)

    vocab = list(model.wv.index_to_key)
    word_freq = Counter()
    for sentence in bootstrap_data:
        word_freq.update(sentence)

    # Update the frequency of special tokens to a high value
    for token in special_tokens:
        word_freq[token] = 99999  # set to some high number


    model.build_vocab_from_freq(word_freq)  # Build vocabulary from updated word frequencies


    model.train(bootstrap_data, total_examples=len(bootstrap_data), epochs=1)
    end_time = datetime.now()

    correlation, p_value = evaluate_model(model)

    log_message = f"Bootstrapping Iteration: {i}, Time: {start_time} to {end_time}, Correlation: {correlation}, p-value: {p_value}"
    logging.info(log_message)

    model.save(f"newest_bootstrap_word2vec_model_{i}.model")
