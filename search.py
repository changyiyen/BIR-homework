#!/usr/bin/python

# Import necessary packages
## Installed
import flask
import werkzeug.utils
import nltk
import numpy as np
import pandas as pd

## Builtins
import json
import os
import os.path
from collections import OrderedDict

# Set upload folder
UPLOAD_FOLDER = 'corpus'
ALLOWED_EXTENSIONS = {'xml'}

# Define Flask app
#app = flask.Flask(__name__, static_url_path="/corpus_kd")
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'falconpunch'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+[a-zA-Z-][a-zA-Z0-9-]+').tokenize
stemmer = nltk.stem.porter.PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

TOPCOUNT = 10

# Load config file
with open("config_tfidf.json", "r", encoding="UTF-8") as conf:
    config = json.load(conf)
# Load corpus file (technically we just want the metadata for the files in the corpus)
with open(config["corpus"], "r", encoding="UTF-8") as f:
    corpus = json.load(f)
# Load TFIDF files
tfidf_1 = pd.read_csv("tfidf_kd.csv", index_col=0)
tfidf_2 = pd.read_csv("tfidf_bigram_kd.csv", index_col=0)

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
        
        if modelname == "tfidf":
            tfidf = tfidf_1
        if modelname == "tfidf2":
            tfidf = tfidf_2
        
        # Tokenize and clean up the search terms
        searchtokens = tokenizer(searchterm)
        searchtokens_stemmed = [stemmer.stem(token) for token in searchtokens if token not in stopwords]
        
        if modelname == "tfidf2":
            bigram_tokens = list()
            for i in range(len(searchtokens_stemmed)):
                bigram_tokens.append(" ".join(searchtokens_stemmed[i:i+2]))
            searchtokens_stemmed = bigram_tokens
        
        # Sum the TFIDF values from table
        query = pd.core.series.Series(np.zeros(len(tfidf)), index=tfidf.index)
        for token in searchtokens_stemmed:
            try:
                query = query.add(tfidf[token])
            except:
                continue
        query.index = tfidf.index
        query.sort_values(inplace=True, ascending=False)
        # Get top docs
        top = OrderedDict(query.head(TOPCOUNT))
        
        # Get file name and TFIDF sums
        if sum(query) > 0:
            results = [(os.path.basename(file), corpus[file]["title"], corpus[file]["token_count"], top[file]) for file in top]
        else:
            results = list()
        return flask.render_template('results.htm', searchterm=searchterm, results=results, topcount=TOPCOUNT)

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