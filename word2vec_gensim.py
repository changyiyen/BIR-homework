#!/usr/bin/python3

# Reference: https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne/

import argparse
import json
import random

import bs4
import nltk
import nltk.stem.porter

STOP_WORDS = nltk.corpus.stopwords.words()
stemmer = nltk.stem.porter.PorterStemmer()

import pandas
pandas.options.mode.chained_assignment = None 

import numpy as np

import gensim.utils
from gensim.models import word2vec

# Install Scikit with `pip install scikit-learn`
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocess_xml(config):
    """
    Preprocess PubMed XML into training CSV files.
    args: config (a dict (JSON object) containing program settings)
    """
    files = list()
    with open(config["filelist"], "r") as filelist:
        files = filelist.read().splitlines()
    # Split files into training and validation sets using defined training-to-total-filecount ratio
    training_files = set(random.sample(files, round(len(files) * config["trainingratio"])))
    validation_files = set(files) - training_files
    training_sentences = list()
    validation_sentences = list()
    max_training_sent_len = 0
    max_validation_sent_len = 0
    outname = config["trainingcorpus"]
    for file in training_files:
        # Read XML, extract abstract and clean it
        sentence_tokens = list()
        with open(file, "r", encoding="UTF-8") as xmlfile:
            soup = bs4.BeautifulSoup(xmlfile, 'xml')
        try:
            abstracttext = soup.AbstractText.text
            if config["tokenize"] == "stem":
                sentence_tokens.extend([stemmer.stem(token) for token in nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)])
                #print(sentence_tokens)
            else:
                sentence_tokens.extend(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext))
                #print(sentence_tokens)
            for word in sentence_tokens:
                if word in STOP_WORDS:
                    sentence_tokens.remove(word)  
            if len(sentence_tokens) > max_training_sent_len:
                max_training_sent_len = len(sentence_tokens)
            #abstracttext = ",".join(sentence_tokens)
            abstracttext = " ".join(sentence_tokens)
        except:
            abstracttext = ""    
        training_sentences.append(abstracttext)
    with open(outname, 'a') as outfile:
        #print(','.join([str(i) for i in range(max_training_sent_len)]), file=outfile)
        for sentence in training_sentences:
            print(sentence, file=outfile) 
    outname = config["validationcorpus"]
    for file in validation_files:
        # Read XML, extract abstract and clean it
        sentence_tokens = list()
        with open(file, "r", encoding="UTF-8") as xmlfile:
            soup = bs4.BeautifulSoup(xmlfile, 'xml')
        try:
            abstracttext = soup.AbstractText.text
            if config["tokenize"] == "stem":
                sentence_tokens.extend([stemmer.stem(token) for token in nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)])
            else:
                sentence_tokens.extend(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext))
            for word in list(abstracttext):
                if word in STOP_WORDS:
                    abstracttext.remove(word)  
            if len(abstracttext) > max_validation_sent_len:
                max_validation_sent_len = len(abstracttext)
            #abstracttext = ",".join(abstracttext)
            abstracttext = " ".join(abstracttext)
        except:
            abstracttext = ""
        validation_sentences.append(abstracttext)
    with open(outname, 'a') as outfile:
        #print(','.join([str(i) for i in range(max_validation_sent_len)]), file=outfile)
        for sentence in validation_sentences:
            print(sentence, file=outfile)

def clean_dataframe(data):
    """Drop NaNs from data frame"""
    data = data.dropna(how="any")
    return data

def build_corpus(data):
    """
    Creates a list of lists containing words from each sentence
    """
    corpus = []
    for i in range(len(data)):
        sentence = data.iloc[[i]].to_string()
        word_list = [word for word in sentence.split(' ') if not word == '']
        corpus.append(word_list)
    return corpus

def tsne_plot(model, config, wordlist=[], highlight_word=""):
    """Creates t-SNE plot"""
    labels = []
    tokens = []
    if wordlist:
        for word in wordlist:
            tokens.append(model.wv.get_vector(word))
            labels.append(word)
    else:
        for word in model.wv.index_to_key:
            tokens.append(model.wv.get_vector(word))
            labels.append(word)
    # Add the word to be highlighted to the end of the token and label lists
    if highlight_word:
        tokens.append(model.wv.get_vector(highlight_word))
        labels.append(highlight_word)
    tokens = np.asarray(tokens)
    # 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree
    #tsne_model = TSNE(perplexity=100, n_components=2, init='pca', n_iter=2500, random_state=42)
    tsne_model = TSNE(perplexity=100, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:    
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16)) 
    if highlight_word:
        for i in range(len(x)-1):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.scatter(x[-1],y[-1], color="red")
        plt.annotate(labels[-1],
                         xy=(x[-1], y[-1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         color="red")
    else:
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.savefig(config["tsne_png"])
    

def pca_plot(model, config, wordlist=[], highlight_word=""):
    """Creates PCA model and plots it"""
    labels = []
    tokens = []
    if wordlist:
        for word in wordlist:
            tokens.append(model.wv.get_vector(word))
            labels.append(word)
    else:
        for word in model.wv.index_to_key:
            tokens.append(model.wv.get_vector(word))
            labels.append(word)
    # Add the word to be highlighted to the end of the token and label lists
    if highlight_word:
        tokens.append(model.wv.get_vector(highlight_word))
        labels.append(highlight_word)
    tokens = np.asarray(tokens)
    pca_model = PCA(n_components=4)
    new_values = pca_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:    
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16)) 
    if highlight_word:
        for i in range(len(x)-1):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.scatter(x[-1],y[-1], color="red")
        plt.annotate(labels[-1],
                         xy=(x[-1], y[-1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         color="red")
    else:
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.savefig(config["pca_png"])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="An implementation of word2vec using Gensim",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="file containing configurations", default="config_gensim.json")
    parser.add_argument("-s", "--similar", type=str, help="string to search similar words for")
    args = parser.parse_args()

    with open(args.config, "r", encoding="UTF-8") as conf:
        config = json.load(conf)
    
    # Step 0: load training data from XML and build training files
    if config["preprocess_xml"] == "true":
        print("[INFO] Preprocessing PubMed XML files to training file")
        preprocess_xml(config)
    # Step 1: load training file and build corpus
    if config["train_model"] == "true":
        print("[INFO] Building corpus from training file")
        data = pandas.read_csv(config["trainingcorpus"], skip_blank_lines=True)
        #data = clean_dataframe(data)
        corpus = build_corpus(data)
    # Step 2: build models using Gensim
        print("[INFO] Building model")
        if config["model_name"] == "cbow":
            # CBOW
            model = word2vec.Word2Vec(corpus, vector_size=config["vector_size"], window=config["CBOW_N_WORDS"], min_count=config["CBOW_MIN_WORD_FREQUENCY"], workers=config["threads"], alpha=config["learning_rate"], epochs=config["epochs"])
        elif config["model_name"] == "skipgram":
            # Skipgram
            model = word2vec.Word2Vec(corpus, vector_size=config["vector_size"], window=config["SKIPGRAM_N_WORDS"], min_count=config["SKIPGRAM_MIN_WORD_FREQUENCY"], workers=config["threads"], sg=1, alpha=config["learning_rate"], epochs=config["epochs"])
        print("[INFO] Model built, saving...")
        model.save(config["saved_model"])
    if config["validate_model"] == "true":
        print("[INFO] Loading model")
        model = gensim.utils.SaveLoad.load(config["saved_model"])
        # Get top 100 most similar words using cosine similarity
        if args.similar:
            print("[INFO] Showing top 100 words and vectors")
            top_100 = model.wv.most_similar(args.similar, topn=100)
            print(top_100)
        if args.similar:
            print("[INFO] Generating t-SNE plot for designated word")
            top_100_words = [i[0] for i in top_100]
            tsne_plot(model, config, top_100_words, args.similar)
            print("[INFO] Generating PCA plot for designated word")
            pca_plot(model, config, top_100_words, args.similar)
        else:
            print("[INFO] Generating t-SNE plot for model")
            tsne_plot(model, config)
            print("[INFO] Generating PCA plot for model")
            pca_plot(model, config)