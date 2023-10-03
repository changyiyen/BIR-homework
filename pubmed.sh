#!/usr/env sh
# dependencies: XMLStarlet, (wget or curl)
if [ ! -e "`which xml`" ]; then
    echo "[ERROR] XMLStarlet not found in \$PATH."
	exit 1
fi

# downloader preference: wget > curl > fetch
if [ -e "`which wget`" ]; then
    DL="wget"
elif [ -e "`which curl`" ]; then
    DL="curl"
else
    DL="fetch"
fi

if [ `uname -s`=='FreeBSD' ]; then
	SLEEPTIME=`jot -r 1 1 600`
elif [ `uname -s`=='Linux' ]; then
	SLEEPTIME=$((RANDOM % 600))
fi
sleep $SLEEPTIME
## Change this to whereever you would like the results placed
cd /home/cyyen/ncbi-pubmed
baseurl="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Set PubMed as database to search
searchquery_base="esearch.fcgi?db=pubmed&"

## Search query terms; modify these as needed ##
searchquery_term="term=HLA+Antigens[Mesh]+AND+Lupus+Erythematosus+,+Systemic[Mesh]"
searchquery_param="&sort=pubdate&retmax=10"
searchquery=${searchquery_base}${searchquery_term}${searchquery_param}
currentdate=`date -I`

# Get search result XML file
if [ $DL == "wget" ]; then
    wget ${baseurl}${searchquery} -O ${currentdate}_search.xml
elif [ $DL =="curl" ]; then
    curl ${baseurl}${searchquery} -o ${currentdate}_search.xml
else
    fetch ${baseurl}${searchquery} -o ${currentdate}_search.xml
fi
# Parse XML file to get list of PubMed IDs
list=`xml fo -D ${currentdate}_search.xml | xml sel -t -v "eSearchResult/IdList/Id" | tr '\n' ','`
# Fetch abstract files
fetchquery="efetch.fcgi?db=pubmed&id=$list&rettype=abstract&retmode=text"
if [ $DL == "wget" ]; then
    wget ${baseurl}${fetchquery} -O ${currentdate}_fetch.txt
elif [ $DL =="curl" ]; then
    curl ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
else
    fetch ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
fi
bzip2 ${currentdate}_fetch.txt
