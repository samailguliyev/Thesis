#!/bin/bash

# Download the NLTK Python package
wget https://files.pythonhosted.org/packages/f6/1d/d925cfb4f324ede997f6d47bea4d9babba51b49e87a767c170b77005889d/nltk-3.6.2.tar.gz

# Extract the NLTK package
tar -xzvf nltk-3.6.2.tar.gz

# Navigate into the directory
cd nltk-3.6.2

# Install the NLTK package
python3 setup.py install

# Navigate back to original directory
cd ..

# Download nltk_data (punkt tokenizer models)
wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip

# Unzip the downloaded nltk_data package
unzip punkt.zip -d nltk_data/

# Clean up
rm nltk-3.6.2.tar.gz
rm punkt.zip
