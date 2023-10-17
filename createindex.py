#!/usr/bin/python

# Import necessary packages
## Installed
## (NB.: bs4 requires the lxml package for parsing XML)
import bs4
import nltk
import nltk.stem.porter
import matplotlib
import matplotlib.pyplot    
### Consider installing defusedxml for better security
#import defusedxml

## Builtins
import argparse
import json
import os

# Build argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--filelist", help="List of filepaths to index (one on each line)", nargs='?', default="manifest.txt", type=str)
parser.add_argument("-s", "--single", help="Collectively treat all files in corpus as one unit, without recording filenames for each token occurence", action='store_true')
parser.add_argument("-p", "--porter", help="Use Porter stemming for index", action='store_true')
parser.add_argument("-t", "--filetype", help="Filetypes of input files to index", type=str, choices=["twitterjson", "pubmedxml"])
args = parser.parse_args()

# Read list of file paths to index
files = list()
with open(args.filelist, "r") as filelist:
    files = filelist.read().splitlines()

# Create index
if args.filetype == "twitterjson":
    index = dict()
    index['word_index'] = dict()
    index['docstats'] = dict()
    index['tweet_filename'] = dict()
    index['mode'] = ""
    for file in files:
        ## Read JSON
        ## https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
        with open(file, "r", encoding="UTF-8") as jsonfile:
            f = json.load(jsonfile)
        for tweet in f:
            #tokens = nltk.tokenize.word_tokenize(tweet["text"])
            tokens = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(tweet["Text"])
            txt = nltk.Text(tokens)
            ## Frequency distribution of tokens
            fq = nltk.probability.FreqDist(token.lower() for token in tokens)
            for word in fq.keys():
                try:
                    index['word_index'][word]
                except KeyError:
                    index['word_index'][word] = dict()
                index['word_index'][word][tweet["ID"]] = fq[word]
            index['docstats'][tweet['ID']] = dict()
            index['docstats'][tweet['ID']]['charn'] = len(tweet["Text"])
            index['docstats'][tweet['ID']]['wordn'] = len(tokens)
            index['docstats'][tweet['ID']]['sentn'] = len(nltk.sent_tokenize(tweet["Text"]))
            index['tweet_filename'][tweet['ID']] = file
            index['mode'] = "single"
        with open("jsonindex.json", "w") as outfile:
            json.dump(index, outfile)

elif args.filetype =="pubmedxml":
    index = dict()
    index['word_index'] = dict()
    index['docstats'] = dict()
    index['mode'] = ''
    tokens = list()
    if args.porter:
        stemmer = nltk.stem.porter.PorterStemmer()
    if not args.single:
        for file in files:
            ## Read XML
            with open(file, "r", encoding="UTF-8") as xmlfile:
                soup = bs4.BeautifulSoup(xmlfile, 'xml')
            #tokens = nltk.tokenize.word_tokenize(soup.AbstractText.text)
            try:
                abstracttext = soup.AbstractText.text
                tokens = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)
                if args.porter:
                    tokens = [stemmer.stem(token) for token in tokens]
            except:
                abstracttext = ""
                tokens = list()
            txt = nltk.Text(tokens)
            ## Frequency distribution of tokens
            fq = nltk.probability.FreqDist(token.lower() for token in tokens)
            for word in fq.keys():
                try:
                    index['word_index'][word]
                except KeyError:
                    index['word_index'][word] = dict()
                index['word_index'][word][file] = fq[word]
            # Calculate document stats
            index['docstats'][file] = dict()
            index['docstats'][file]['pmid'] = soup.PMID.text
            index['docstats'][file]['charn'] = len(abstracttext)
            index['docstats'][file]['wordn'] = len(tokens)
            index['docstats'][file]['sentn'] = len(nltk.sent_tokenize(abstracttext))
            index['mode'] = 'multi'
    if args.single:
        for file in files:
            ## Read XML
            with open(file, "r", encoding="UTF-8") as xmlfile:
                soup = bs4.BeautifulSoup(xmlfile, 'xml')
            #tokens = nltk.tokenize.word_tokenize(soup.AbstractText.text)
            try:
                abstracttext = soup.AbstractText.text
                if args.porter:
                    tokens.extend([stemmer.stem(token) for token in nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)])
                else:
                    tokens.extend(nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext))
            except:
                abstracttext = ""
            # Calculate document stats
            index['docstats'][file] = dict()
            index['docstats'][file]['pmid'] = soup.PMID.text
            index['docstats'][file]['charn'] = len(abstracttext)
            index['docstats'][file]['wordn'] = len(tokens)
            index['docstats'][file]['sentn'] = len(nltk.sent_tokenize(abstracttext))
            
        fig = matplotlib.pyplot.figure(figsize = (20,8))
        matplotlib.pyplot.gcf().subplots_adjust(bottom=0.15)
        ## Frequency distribution of tokens
        fq = nltk.probability.FreqDist(token.lower() for token in tokens)
        fq.plot(100, title="Most common 100 words in index")
        fig.savefig('freqDist.png')
        for word in fq.keys():
            try:
                index['word_index'][word]
            except KeyError:
                index['word_index'][word] = dict()
            index['word_index'][word] = fq[word]
        index['mode'] = 'single'
    with open("xmlindex.json", "w") as outfile:
        json.dump(index, outfile)
 