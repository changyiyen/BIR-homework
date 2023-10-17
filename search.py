#!/usr/bin/python

# Import necessary packages
## Installed
import bs4
import flask
import werkzeug.utils
import nltk
### Incongruent import method here due to NLTK packaging issues (see https://stackoverflow.com/questions/33398282/attributeerror-module-object-has-no-attribute-scores)
from nltk.metrics import edit_distance

## Builtins
import argparse
import glob
import json
import os
import os.path

# Set upload folder
UPLOAD_FOLDER = 'corpus'
ALLOWED_EXTENSIONS = {'json', 'xml'}

# Define Flask app
app = flask.Flask(__name__, static_url_path="/corpus")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'falconpunch'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def main():
    ## Search page on GET
    if flask.request.method == 'GET':
        corpus = glob.glob('corpus/*')
        return flask.render_template('search.htm', corpus=corpus)

    ## Results page on POST
    if flask.request.method == 'POST':
        # Generate search results
        results = list()
        index = dict()
        ## Read index
        if flask.request.form.get('searchtype') == 'tweet':
            with open('jsonindex.json', 'r') as tweetindex:
                index = json.load(tweetindex)
        if flask.request.form.get('searchtype') == 'pubmedxml':
            with open('xmlindex.json', 'r') as pubmedindex:
                index = json.load(pubmedindex)
        if flask.request.form.get('searchtype') == 'pmid':
            with open('xmlindex.json', 'r') as pubmedindex:
                index = json.load(pubmedindex)
        
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
                
        
        if flask.request.form.get('searchtype') == 'pmid':
            result = dict()
            f = ""
            for file in index['docstats']:
                if index['docstats'][file]['pmid'] == flask.request.form.get('searchterm'):
                    f = file
                    break
            result['context'] = ""
            result['hits'] = ""
            try:
                result['docstats'] = index['docstats'][f]
            except:
                result['docstats'] = ""
            results.append(result)

        searchterm = flask.request.form.get('searchterm')
        ## should we stem the searchterm?

        ## Search term not found in index: try fuzzy search (get closest term)
        if searchterm not in index['word_index'].keys() and index['mode'] == 'multi':
            distance = list()
            for token in index['word_index'].keys():
                distance.append((token, edit_distance(searchterm, token)))
            distance = sorted(distance, key=lambda x:x[1], reverse=False)
            searchterm = distance[0][0]

        ##### Search term found in index #####
        if searchterm in index['word_index'].keys() and index['mode'] == 'multi':
            for id, hits in index['word_index'][searchterm].items():
                result = dict()
                # Tweets
                if flask.request.form.get('searchtype') == 'tweet':
                    result['filename'] = id
                    result['url'] = index['tweet_filename'][id]
                    with open(result['url'], 'r') as f:
                        result['context'] = f.read().replace(searchterm, '<mark>'+searchterm+'</mark>')
                # Pubmed XML
                if flask.request.form.get('searchtype') == 'pubmedxml':
                    result['filename'] = os.path.basename(id)
                    result['url'] = id
                    with open(result['url'], 'r', encoding="UTF-8") as f:
                        abstract = bs4.BeautifulSoup(f, 'xml').AbstractText.text
                        result['context'] = abstract.replace(searchterm, '<mark>'+searchterm+'</mark>')
                result['hits'] = hits
                result['docstats'] = index['docstats'][result['url']]
                results.append(result)
        ##### Search term *not* found in index: full text search :( #####
        else:
            if flask.request.form.get('searchtype') == 'tweet':
                filelist = glob.glob('corpus/*.json')
                for file in filelist:
                    result = dict()
                    with open(file, 'r') as tweetfile:
                        tweets = json.load(tweetfile)
            if flask.request.form.get('searchtype') == 'pubmedxml':
                filelist = glob.glob('corpus/*.xml')
                for file in filelist:
                    result = dict()
                    hits = 0
                    with open(file, 'r') as pubmedfile:
                        try:
                            abstract = str(bs4.BeautifulSoup(pubmedfile, 'xml').AbstractText.text)
                        except:
                            abstract = ""
                    if abstract.count(searchterm) > 0:
                        result['filename'] = file
                        result['url'] = 'corpus/' + file
                        # Need to do something better for context, instead of just dumping the whole abstract
                        result['context'] = abstract.replace(searchterm, '<mark>'+searchterm+'</mark>')
                        result['hits'] = abstract.count(searchterm) 
                        result['docstats'] = index['docstats'][file]
                        results.append(result)
        results.sort(reverse=True, key=lambda x: x['hits'])
        return flask.render_template('results.htm', searchterm=searchterm, results=results)

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