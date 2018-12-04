import sys
from ner.network import NER
from ner.corpus import Corpus
import json
from ner.utils import md5_hashsum, download_untar
from glob import glob
from ner.utils import tokenize, lemmatize
import os

# This script provides command line interface for job descriptions
# Just run something like command below in terminal
"Industry, Job title, Department, and Skills"

# Load network params
with open('Nandhakumar/AmitSharma.json') as f:
    network_params = json.load(f)


corpus = Corpus(dicts_filepath='model/Nandhakumar.txt')

network = NER(corpus, verbouse=False, pretrained_model_filepath='model/ner_model', **network_params)


def print_predict(sentence):
    # Split sentence into tokens
    tokens = tokenize(sentence)

    # Lemmatize every token
    tokens_lemmas = lemmatize(tokens)

    tags = network.predict_for_token_batch([tokens_lemmas])[0]
    for token, tag in zip(tokens, tags):
        print(token, tag)


for query in sys.stdin:
    print_predict(query)
