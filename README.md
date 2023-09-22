# BIR-homework
Homework code repository for Biological Information Retrieval course at NCKU

## Dependencies

Dependencies are:

- Flask (for web interface)
- NLTK (for NLP operations)
- Beautiful Soup and lxml (for XML parsing)

Install with: `pip install Flask nltk beautifulsoup4 lxml`

## Usage

- Place files to be indexed (PubMed XML files or Twitter tweet files) into the `corpus` directory (files may also be uploaded to the corpus directory using the web interface)
- Create manifest file (a file containing the relative paths of the text files to be indexed, one on each line, e.g., `ls corpus/* > manifest.txt`)
- Build index file using `createindex.py` (e.g., `python createindex.py -l manifest.txt -t pubmedxml`)
- Start web interface with `python search.py` and navigate to http://127.0.0.1:5000
