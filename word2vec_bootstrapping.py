
import os
import json
import time
from collections import Counter
from scipy.stats import spearmanr
import pandas as pd
from gensim.models import Word2Vec
import random

print('new version')





import pandas as pd
from datetime import datetime

word2vec_data = []  # Assuming this is your initialized list for word2vec data

keywords = ['tinder', 'zoom', 'telegram', 'twitch', 'discord', 'teams', 'slack', 'stripe', 'snap', 'swift', 'bumble', 'medium', 'hinge', 'notion', 'signal', 'azure', 'echo', 'alexa', 'prime', 'edge', 'meta', 'lightning', 'eats', 'spark', 'hana', 'ring', 'square', 'corona', 'vegan', 'vegans', 'veganism']

# Generate special tokens
special_tokens = []
years = range(2009, 2023)
for keyword in keywords:
    for year in years:
        special_tokens.append(f"{keyword}{year}")


chunk_size = 3_000_000  # You can adjust this size based on your available memory

for i, chunk in enumerate(pd.read_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe.csv", sep='\t', chunksize=chunk_size)):

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


# Generate special tokens
special_tokens = []
# keywords = ['tinder', 'zoom', 'telegram', 'vegan', 'vegans', 'veganism', 'vegetarian', 'vegetarianism', 'vegetarians','slack', 'teams',
#              'discord', 'twitch']
years = range(2009, 2023)
for keyword in keywords:
    for year in years:
        special_tokens.append(f"{keyword}{year}")



# Function for evaluation
def evaluate_model(model, simlex_data, vocab):
    simlex_pairs = []
    model_pairs = []
    for i, row in simlex_data.iterrows():
        word1, word2, score = row['word1'], row['word2'], row['SimLex999']
        if word1 in vocab and word2 in vocab:
            simlex_pairs.append(score)
            model_pairs.append(model.wv.similarity(word1, word2))
    return spearmanr(simlex_pairs, model_pairs)

# Function to divide the dataset into smaller chunks
def chunk_data(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# ... [Your existing imports and functions here]

# Define the number of bootstrapping iterations
n_bootstrap_iterations = 5

for bootstrap_iter in range(1, n_bootstrap_iterations + 1):
    print(f"=== BOOTSTRAP ITERATION {bootstrap_iter} ===")

    # Get a random sample of 50% of the data
    random.seed(bootstrap_iter)  # Use the loop iteration as the seed for reproducibility
    bootstrap_sample = random.sample(word2vec_data, int(len(word2vec_data) * 0.5))

    # Initialize Word2Vec model
    model = Word2Vec(
        vector_size=300,
        window=2,
        min_count=150,
        sample=0.005,
        negative=20,
        alpha=0.025,
        sg=1,
        workers=64,
        seed=1234,
        compute_loss=True,
        hs = 0  # Added this line to compute loss
    )
    # Check if vocab.json and word_freq.json already exist
    if os.path.exists('vodfdfdfcab.json') and os.path.exists('wordfdfsdfd_freq.json'):
        print("Reading existing vocabulary and word frequencies...")
        with open('vocab.json', 'r') as f:
            vocab = json.load(f)
        with open('word_freq.json', 'r') as f:
            word_freq = json.load(f)
            
        # Update the frequency of special tokens to a high value
        for token in special_tokens:
            word_freq[token] = 999999  # set to some high number

        # Build vocabulary from updated word frequencies
        model.build_vocab_from_freq(word_freq)
        model.build_vocab([special_tokens], update=True)  # Update just to be sure

        # Save the updated word frequency JSON
        with open('word_freq.json', 'w') as f:
            json.dump(word_freq, f)

    else:
        print("Building vocabulary...")
        vocab_start_time = time.time()

        # Assuming word2vec_data is already loaded
        # model.build_vocab(word2vec_data, progress_per=1000)
        vocab = list(model.wv.index_to_key)
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f)

        word_freq = Counter()
        for sentence in word2vec_data:
            word_freq.update(sentence)

        # Update the frequency of special tokens to a high value
        for token in special_tokens:
            word_freq[token] = 99999  # set to some high number

        # Save the updated word frequency JSON
        with open('word_freq.json', 'w') as f:
            json.dump(word_freq, f)

        model.build_vocab_from_freq(word_freq)  # Build vocabulary from updated word frequencies
        # model.build_vocab([special_tokens], update=True)  # Update just to be sure
        vocab_build_time = time.time() - vocab_start_time


    # Load SimLex-999 dataset
    simlex_data = pd.read_csv('/home/ma/ma_ma/ma_sguliyev/New_thesis_repository/Data/SimLex-999/SimLex-999.txt', sep='\t')

    best_correlation = -1
    best_epoch = 0
    try:
        metrics = {"vocab_build_time": vocab_build_time, "epochs": []}
    except:
        metrics = {"epochs": []}

    # ... [Your existing vocabulary building logic here]

    # Partial training
    for epoch in range(1, 2):  # 10 epochs
        # print(f'Epoch {epoch}')
        start_time = time.time()
        for i, data_chunk in enumerate(chunk_data(bootstrap_sample)):
            if i % 1000 == 0:
                print(f"chunk {i}")
            model.train(data_chunk, total_examples=len(data_chunk), epochs=1)
            chunk_loss = model.get_latest_training_loss()  # Added this line to get the loss

        correlation, _ = evaluate_model(model, simlex_data, vocab)
        print(f"Epoch: {epoch}, Correlation: {correlation}")
        end_time = time.time() - start_time
        print(f"Epoch: {epoch}, Correlation: {correlation}, Time: {end_time} seconds")



        vocab_words = set(model.wv.index_to_key)  # Convert vocabulary to a set for faster lookup

        all_present = all(token in vocab_words for token in special_tokens)

        if all_present:
            print("All special tokens are present in the model vocabulary.")
        else:
            missing_tokens = [token for token in special_tokens if token not in vocab_words]
            print(f"The following special tokens are missing: {missing_tokens}")


        model.save(f"new_bootstrap_model_{bootstrap_iter}_epoch_{epoch}.model")

    print(f"=== END BOOTSTRAP ITERATION {bootstrap_iter} ===")
