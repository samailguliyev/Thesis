import os
import logging
import pandas as pd
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd

import pandas as pd
import re
import contractions
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from nltk.corpus import stopwords
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import bigrams
from collections import Counter
import re
import pycld2 as cld2
import time
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')


# Set up logging
logging.basicConfig(filename='preprocessing_AskWomen.log', level=logging.INFO)

# Load stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Handling words with hyphens
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    
    # Removing links
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Removing HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Opening contractions
    text = contractions.fix(text)
    
    # Removing numbers and special characters (excluding underscore)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    
    # Keeping only English characters (assuming ASCII range for English characters)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    
    # Removing stopwords and single characters
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words and len(word) > 1])
    
    return text

def preprocess_batch(texts):
    return [preprocess_text(text) for text in texts]

def process_file(filepath, output_dir):
    filename = os.path.basename(filepath)
    logging.info(f'Processing file: {filename}')
    
    df = pd.read_json(filepath, lines=True)
    logging.info(f'Original DataFrame length: {len(df)}')
    
    df = df[df['body'] != '[deleted]']
    
    n_batches = 3
    batch_size = len(df) // n_batches
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    preprocessed_batches = Parallel(n_jobs=-1)(delayed(preprocess_batch)(batch['body'].tolist()) for batch in batches)
    preprocessed_texts = [text for batch in preprocessed_batches for text in batch]
    
    df["preprocessed_body"] = preprocessed_texts
    df = df[df['preprocessed_body'].apply(lambda x: len(x.split()) >= 2)]
    logging.info(f'Preprocessed DataFrame length: {len(df)}')
    
    output_filepath = os.path.join(output_dir, f'preprocessed_{filename}')
    df.to_json(output_filepath, lines=True, orient='records')
    logging.info(f'Saved preprocessed data to: {output_filepath}')

def main():
    input_dir = '/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskWomen_download/AskWomen_per_year_comments_14M_adjusted'
    output_dir = '/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskWomen_download/AskWomen_per_year_comments_14M_adjusted_preprocessed_2'  # Change this to your desired output directory
    
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if filepath.endswith('.json') and os.path.isfile(filepath):
            process_file(filepath, output_dir)

if __name__ == '__main__':
    main()
