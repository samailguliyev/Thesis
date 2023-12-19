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
logging.basicConfig(filename='Big_AskGender_model_training.log', level=logging.INFO)

keywords = ['tinder', 'zoom', 'telegram', 'twitch', 'discord',
             'teams', 'slack', 'stripe', 'snap', 'swift',
             'bumble', 'medium', 'hinge', 'notion', 'signal', 'azure', 
             'echo', 'alexa', 'prime', 'edge', 'meta', 'lightning', 'eats', 
             'spark', 'corona', 'vegan', 'vegans', 'veganism']

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
    "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps_restored/AskWomen_download/AskWomen_per_year_comments_splitted/original_dataframe.csv",
     "askwomen"
    ) + process_dataframe(
        "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps_restored/AskMen_download/AskMen_per_year_comments_splitted/original_dataframe.csv",
         "askmen")


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
    'min_count': 150,
    'sample': 0.005,
    'negative': 30,
    'alpha': 0.025,
    'sg': 1,
    'workers': 512,
    'batch_words': 100,
    'hs': 0
}

# List of sampling percentages
sampling_percentages = [1, 100 , 5, 10, 25, 50]

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
    adjusted_min_count = int(150 * (percentage / 100))
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

    log_message = f"Sampling Percentage: {percentage}%, Time: {start_time} to {end_time}, Correlation: {correlation}, p-value: {p_value}"
    logging.info(log_message)

    model.save(f"sampled_gendered_word2vec_model_{percentage}_percent.model")

print("Training completed for all sampling percentages.")
