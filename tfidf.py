#!/usr/bin/python3

# Usage: word2vec.py [-h] [-c configfile]

# Import builtins
import argparse
import json
import math
import os
import random

# Import external packages
import bs4
import nltk
import numpy as np
import nltk.stem.porter
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Parse XML files and write abstract to text file
def parsexml(config):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+[a-zA-Z-][a-zA-Z0-9-]+').tokenize
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
        
    files = list()
    with open(config["filelist"], "r") as filelist:
        files = filelist.read().splitlines()
    
    index = dict()
    outname = config["corpus"]
    for file in files:
        index[file] = dict()
        ## Read XML
        with open(file, "r", encoding="UTF-8") as xmlfile:
            soup = bs4.BeautifulSoup(xmlfile, 'xml')
        try:
            abstracttext = soup.AbstractText.text
            tokens = tokenizer(abstracttext)
        except:
            abstracttext = ""
            tokens = list()
        try:
            title = soup.ArticleTitle.text
        except:
            title = ""
        tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
        index[file]["contents"] = ' '.join(tokens)
        index[file]["title"] = title
        index[file]["token_count"] = len(tokens)

    with open(outname, 'w') as outfile:
        json.dump(index, outfile)

    print("[INFO] XML files now parsed into a corpus file with each document on its own line.")
    print("[INFO] Tokens are separated with a single space.")
    print("[INFO] Please change the mode in the config file from 'raw' to 'tfidf' to continue.")
    print("[INFO] Program will now exit.")
    exit(0)

# Step 2a: build table of tf-idf values for each token for each document
def tfidf(config):
    with open(config["corpus"], "r", encoding="UTF-8") as f:
        corpus_raw = json.load(f)
    # Ignore empty files
    corpus = dict()
    for file in corpus_raw.keys():
        if len(str(corpus_raw[file]["title"])) > 0 and len(str(corpus_raw[file]["contents"])) > 0:
            corpus[file] = dict()
            corpus[file]["title"] = corpus_raw[file]["title"]
            corpus[file]["contents"] = corpus_raw[file]["contents"]

    # Corpus lookup in fixed order
    corpus_filenames = tuple(corpus.keys())
    corpus_titles = [corpus[i]["title"] for i in corpus_filenames]
    corpus_documents = [ corpus[i]["contents"] for i in corpus_filenames]

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    # The above can be combined to a single call to TfidfVectorizer()
    #vectorizer = TfidfVectorizer()
    # NB.: TfidfVectorizer will perform L2 normalization by default!
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_documents))
    df_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=corpus_filenames)
    print("[INFO] Writing TFIDF")
    with open(config["tfidf_output"], "w") as f:
        df_tfidf.to_csv(f, index=True)

# Step 2b: build table of tf-idf values for bigrams
def tfidf_bigram(config):
    with open(config["corpus"], "r", encoding="UTF-8") as f:
        corpus_raw = json.load(f)
    # Ignore empty files
    corpus = dict()
    for file in corpus_raw.keys():
        if len(str(corpus_raw[file]["title"])) > 0 and len(str(corpus_raw[file]["contents"])) > 0:
            corpus[file] = dict()
            corpus[file]["title"] = corpus_raw[file]["title"]
            corpus[file]["contents"] = corpus_raw[file]["contents"]

    # Corpus lookup in fixed order
    corpus_filenames = tuple(corpus.keys())
    corpus_titles = [corpus[i]["title"] for i in corpus_filenames]
    corpus_documents = [ corpus[i]["contents"] for i in corpus_filenames]

    vectorizer = CountVectorizer(ngram_range=(2,2))
    transformer = TfidfTransformer()
    # The above can be combined to a single call to TfidfVectorizer()
    #vectorizer = TfidfVectorizer()
    # NB.: TfidfVectorizer will perform L2 normalization by default!
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_documents))
    df_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=corpus_filenames)
    print("[INFO] Writing bigram TFIDF")
    with open(config["tfidf_bigram_output"], "w") as f:
        df_tfidf.to_csv(f, index=True)

# Step 2c: build table of modified td-idf for sentences
# Step 2d: TFIDF from scratch
def tfidf_0(config):
    with open(config["corpus"], "r", encoding="UTF-8") as f:
        corpus_raw = json.load(f)
    # Ignore empty files
    corpus = dict()
    for file in corpus_raw.keys():
        if len(str(corpus_raw[file]["title"])) > 0 and len(str(corpus_raw[file]["contents"])) > 0:
            corpus[file] = dict()
            corpus[file]["title"] = corpus_raw[file]["title"]
            corpus[file]["contents"] = corpus_raw[file]["contents"]

    # Corpus lookup in fixed order
    corpus_filenames = tuple(corpus.keys())
    corpus_titles = [corpus[i]["title"] for i in corpus_filenames]
    corpus_documents = [ corpus[i]["contents"] for i in corpus_filenames]

    token_dict = dict()
    
    for i in range(len(corpus_filenames)):
        document = corpus_documents[i].split()
        for token in document:
            if token not in token_dict:
                token_dict[token] = dict()
            if corpus_filenames[i] not in token_dict[token]:
                token_dict[token][corpus_filenames[i]] = 0
            token_dict[token][corpus_filenames[i]] += 1
    tokens = token_dict.keys()
    tfidf_table = pd.DataFrame(np.zeros((len(token_dict),len(corpus_filenames))), columns=corpus_filenames, index=token_dict.keys())
    print(token_dict)
    # TF
    for token in tokens:
        for filename in token_dict[token]:
            tfidf_table[filename][token] = token_dict[token][filename] / len(corpus[filename]["contents"])
    # IDF
    for token in tokens:
        for filename in token_dict[token]:
            tfidf_table[filename][token] *= math.log(len(corpus_filenames) / len(token_dict[token]))
    tfidf_table = tfidf_table.transpose()
    print("[INFO] Writing handbuilt TFIDF")
    with open(config["tfidf_0_output"], "w") as f:
        tfidf_table.to_csv(f, index=True)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFIDF calculation from scratch and using Sci-kit",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="file containing configurations", default="config_tfidf.json")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="UTF-8") as conf:
        config = json.load(conf)

    if config["mode"] == "raw":
        parsexml(config)
    if config["mode"] == "tfidf":
        tfidf(config)
    if config["mode"] == "tfidf2":
        tfidf_bigram(config)
    if config["mode"] == "tfidf_0":
        tfidf_0(config)
    
    