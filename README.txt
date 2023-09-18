# Manifest file sample

./12345678.json
./23456789.json
./34567890.json

# Document sample (JSON)

...

# Document sample (XML)

...

# Index file sample

...

# Aggregated index file sample

{
  'word_index': {
    'word 1': {'filename_or_tweetid_1': 1, 'filename_or_tweetid_1': 2},
    'word 2': {'filename_or_tweetid_3': 2},
    ...
  },
  'docstats': {
    ''filename_or_tweetid_1'': {'charnum': 1234, 'wordnum': 123, 'sentnum': 12},
    ...
  }
  'tweet_filename': {
    'tweetid1': 'filename1',
    'tweetid2': 'filename2'
  }
}

* * *

Program logic:
  1. POST search term to Flask from search UI
  2. Search through word index first
  3. If not present in index: regular full text search (grep?)
  4. Return search results (may use <mark></mark> to mark search entry)
Extras:
  - stemming