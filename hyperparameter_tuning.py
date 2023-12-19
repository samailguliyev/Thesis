# from gensim.models import Word2Vec
# import json
# import pandas as pd
# import logging
# from scipy.stats import spearmanr
# from datetime import datetime
# from collections import Counter

# # Initialize logging
# logging.basicConfig(filename='new_hyperparameter_tuning.log', level=logging.INFO)

# # Read original DataFrame and sample 10%
# df = pd.read_csv(
# "/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe.csv", sep='\t')
# df_sample = df.sample(frac=0.1, random_state=42)

# # Prepare data for Word2Vec
# word2vec_data = [line.split() for line in df_sample['preprocessed_body'].dropna()]

# # Read SimLex dataset
# simlex_data = pd.read_csv('/home/ma/ma_ma/ma_sguliyev/New_thesis_repository/Data/SimLex-999/SimLex-999.txt', sep='\t')

# # Evaluation function
# def evaluate_model(model):
#     simlex_pairs = []
#     model_pairs = []
#     for i, row in simlex_data.iterrows():
#         word1, word2, score = row['word1'], row['word2'], row['SimLex999']
#         if word1 in model.wv and word2 in model.wv:
#             simlex_pairs.append(score)
#             model_pairs.append(model.wv.similarity(word1, word2))
#     return spearmanr(simlex_pairs, model_pairs)

# # Hyperparameter options
# params = {
#     'min_count': [1, 10, 50, 100, 200],
#     'vector_size': [50, 100, 200, 300],
#     'window': [2, 5, 7, 10, 15],
#     'sample': [0.001, 0.005, 0.01, 0.1],
#     'negative': [2, 5, 10, 15, 20],
#     'alpha': [0.01, 0.015, 0.025, 0.035, 0.05],
#     'batch_words': [25, 50, 100, 500, 1000, 5000, 10000],
#     'hs': [0, 1]
# }

# # Initialize results list
# results = []

# # Flatten the corpus for word frequency calculation
# flattened_corpus = [word for sentence in word2vec_data for word in sentence]

# # Perform hyperparameter search
# for param in params.keys():
#     best_value = None
#     best_correlation = None

#     for value in params[param]:
#         try:
#             # Set default parameters
#             default_params = {
#                 'vector_size': 300,
#                 'window': 5,
#                 'min_count': 1,
#                 'sample': 0.01,
#                 'negative': 10,
#                 'alpha': 0.025,
#                 'sg': 1,
#                 'workers': 64,
#                 'batch_words': 1000,
#                 'hs': 0
#             }

#             # Update the parameter being tested
#             default_params[param] = value

#             # Build vocab and word frequency if min_count is the parameter being tested
#             if param == 'min_count':
#                 word_freq = Counter([word for word, count in Counter(flattened_corpus).items() if count >= value])
#                 with open(f'word_freq_min_count_{value}.json', 'w') as f:
#                     json.dump(word_freq, f)

#             # Initialize and train the model
#             start_time = datetime.now()
#             model = Word2Vec(**default_params)
#             if param == 'min_count':
#                 model.build_vocab_from_freq(word_freq)
#             else:
#                 model.build_vocab(word2vec_data)
#             model.corpus_count = len(word2vec_data)
#             model.train(word2vec_data, total_examples=model.corpus_count, epochs=1)
#             end_time = datetime.now()

#             # Evaluate the model
#             correlation, p_value = evaluate_model(model)

#             # Log the results
#             log_message = f"Time: {start_time} to {end_time}, Parameters: {default_params}, Correlation: {correlation}, p-value: {p_value}"
#             logging.info(log_message)

#             # Update best values
#             if best_correlation is None or correlation > best_correlation:
#                 best_correlation = correlation
#                 best_value = value

#         except Exception as e:
#             logging.error(f"Error with parameters {default_params}: {e}")

#     # Log the best value for the current parameter
#     logging.info(f"Best value for {param}: {best_value} with correlation: {best_correlation}")

# # Save the results
# with open('hyperparameter_results.json', 'w') as f:
#     json.dump(results, f)



from gensim.models import Word2Vec
import json
import pandas as pd
import logging
from scipy.stats import spearmanr
from datetime import datetime
from collections import Counter

# Initialize logging
logging.basicConfig(filename='new_hyperparameter_tuning.log', level=logging.INFO)

# Read the original DataFrame and sample 10%
df = pd.read_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskReddit_download/Askreddit_per_year_comments_splitted/original_dataframe.csv", sep='\t')
df_sample = df.sample(frac=0.1, random_state=42)

# Prepare data for Word2Vec
word2vec_data = [line.split() for line in df_sample['preprocessed_body'].dropna()]

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

# Hyperparameter options
params = {
    'min_count': [1, 10, 50, 100, 200, 400],
    'vector_size': [50, 100, 200, 300],
    'window': [2, 5, 7, 10, 15],
    'sample': [0.001, 0.005, 0.01, 0.1],
    'negative': [2, 5, 10, 15, 20],
    'alpha': [0.01, 0.015, 0.025, 0.035, 0.05],
    'batch_words': [25, 50, 100, 500, 1000, 5000, 10000],
    'hs': [0, 1]
}

# Initialize best_params dictionary
best_params = {
    'vector_size': 300,
    'window': 5,
    'min_count': 1,
    'sample': 0.01,
    'negative': 10,
    'alpha': 0.025,
    'sg': 1,
    'workers': 64,
    'batch_words': 1000,
    'hs': 0
}

# Flatten the corpus for word frequency calculation
flattened_corpus = [word for sentence in word2vec_data for word in sentence]

# Initialize word_freq dictionary
word_freq_dict = {}

# Perform hyperparameter search
for param in params.keys():
    best_value = None
    best_correlation = None

    for value in params[param]:
        try:
            # Update the parameter being tested
            best_params[param] = value

            # Build vocab and word frequency if min_count is the parameter being tested
            if param == 'min_count':
                if value in word_freq_dict:
                    word_freq = word_freq_dict[value]
                else:
                    word_freq = {word: count for word, count in Counter(flattened_corpus).items() if count >= value}
                    word_freq_dict[value] = word_freq
                    with open(f'word_freq_min_count_{value}.json', 'w') as f:
                        json.dump(word_freq, f)

            # Initialize and train the model
            start_time = datetime.now()
            model = Word2Vec(**best_params)
            if param == 'min_count':
                model.build_vocab_from_freq(word_freq)
            else:
                model.build_vocab(word2vec_data)
            model.corpus_count = len(word2vec_data)
            model.train(word2vec_data, total_examples=model.corpus_count, epochs=1)
            end_time = datetime.now()

            # Evaluate the model
            correlation, p_value = evaluate_model(model)

            # Log the results
            log_message = f"Time: {start_time} to {end_time}, Parameters: {best_params}, Correlation: {correlation}, p-value: {p_value}"
            logging.info(log_message)

            # Update best values
            if best_correlation is None or correlation > best_correlation:
                best_correlation = correlation
                best_value = value

        except Exception as e:
            logging.error(f"Error with parameters {best_params}: {e}")

    # Update the best value for the current parameter
    best_params[param] = best_value
    logging.info(f"Best value for {param}: {best_value} with correlation: {best_correlation}")
