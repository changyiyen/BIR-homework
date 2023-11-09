#!/usr/bin/python

# Import necessary packages
## Installed
import flask
import gensim
import werkzeug.utils

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Builtins
import os
import os.path

# Set upload folder
UPLOAD_FOLDER = 'corpus'
ALLOWED_EXTENSIONS = {'xml'}

# Define Flask app
#app = flask.Flask(__name__, static_url_path="/corpus")
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'falconpunch'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def tsne_plot(model, wordlist=[], highlight_word=""):
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
    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=2500)
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
    plt.savefig("static/tsne.png")

def pca_plot(model, wordlist=[], highlight_word=""):
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
    plt.savefig("static/pca.png")


@app.route('/', methods=['GET','POST'])
def main():
    ## Search page on GET
    if flask.request.method == 'GET':
        return flask.render_template('search.htm')

    ## Results page on POST
    if flask.request.method == 'POST':
        # Generate search results
        results = list()

        '''
        ## TODO: perform boolean operations on queries
        ## TODO: move searches to their own functions so we can call them and perform boolean operations on returned results
        query = flask.request.form.get('searchterm')
        query = query.replace('AND', '&')
        query = query.replace('OR', '|')
        query = query.replace('NOT', '-')
        querylist = query.split()
        for i in querylist:
            if i in ['&', '|', '-']:
                continue
            else:
                ## replace search term with set of results
                pass
        '''

        searchterm = flask.request.form.get('searchterm')
        modelname = flask.request.form.get('searchtype')
        topwordcount = flask.request.form.get('topwordcount')
        print(flask.request.form)
        if modelname == "als_cbow":
            model = gensim.utils.SaveLoad.load("word2vec_gensim_cbow_als.model")
        elif modelname == "sle_cbow":
            model = gensim.utils.SaveLoad.load("word2vec_gensim_cbow_sle.model")
        try:
            #results = model.wv.most_similar(searchterm)
            # Get top 100 most similar words using cosine similarity
            results = model.wv.most_similar(searchterm, topn=int(topwordcount))
            results_words = [i[0] for i in results]
            tsne_plot(model, results_words, searchterm)
            pca_plot(model, results_words, searchterm)
        except KeyError:
            results = []
        return flask.render_template('results.htm', searchterm=searchterm, results=results, topwordcount=topwordcount)

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            flask.flash('No file part!')
            return flask.redirect(flask.request.url)
        file = flask.request.files['file']
        if file.filename == '':
            flask.flash('No selected file!')
            return flask.redirect(flask.request.url)
        if file and allowed_file(file.filename):
            filename = werkzeug.utils.secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return flask.redirect('/')
    return flask.redirect('/')

if __name__ == '__main__':
    app.run(debug=True)