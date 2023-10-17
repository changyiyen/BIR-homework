#!/usr/env sh
# initial version: 2022-08-08
# current version: 2023-10-12
# dependencies: XMLStarlet, (wget or curl)

## Change this to whereever you would like the results placed
DLDIR="/home/cyyen/ncbi-pubmed"

usage() { echo "Usage: $0 [-m]" 1&>2; exit 1; }

while getopts ":m" option; do
    case "${option}" in
        m)
            m=true ;;
	?)
            usage ;;
    esac
done

if [ ! -e "`which xml`" ]; then
    echo "[ERROR] XMLStarlet not found in \$PATH."
	exit 1
fi

echo "[INFO] XMLStarlet found."

# downloader preference: wget > curl > fetch
if [ -e "`which wget`" ]; then
    DL="wget"
elif [ -e "`which curl`" ]; then
    DL="curl"
else
    DL="fetch"
fi

echo "[INFO] Setting downloader to $DL based on installed options."

# RANDOM not available on FreeBSD; use jot(1) instead
if [ `uname -s`=='FreeBSD' ]; then
	SLEEPTIME=`jot -r 1 1 600`
elif [ `uname -s`=='Linux' ]; then
	SLEEPTIME=$((RANDOM % 600))
fi
sleep $SLEEPTIME

cd $DLDIR
echo "[INFO] Changed working directory to $DLDIR."

baseurl="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Set PubMed as database to search
searchquery_base="esearch.fcgi?db=pubmed&"

## Search query terms; modify these as needed ##
#searchquery_term="term=HLA+Antigens[Mesh]+AND+Lupus+Erythematosus+,+Systemic[Mesh]"
searchquery_term="term=Wounds+,+Gunshot[Mesh]"
searchquery_param="&sort=pubdate&retmax=100"
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
list=`xml fo -D ${currentdate}_search.xml | xml sel -t -v "eSearchResult/IdList/Id"`
if [ "$m" = true ]; then
    for i in $list; do
        fetchquery="efetch.fcgi?db=pubmed&id=$i&rettype=pubmed&retmode=text"
        if [ $DL == "wget" ]; then
            wget ${baseurl}${fetchquery} -O ${i}_fetch.xml
        elif [ $DL =="curl" ]; then
            curl ${baseurl}${fetchquery} -o ${i}_fetch.xml
        else
            fetch ${baseurl}${fetchquery} -o ${i}_fetch.xml
        fi
    done
else
    list=`xml fo -D ${currentdate}_search.xml | xml sel -t -v "eSearchResult/IdList/Id" | tr '\n' ','`
    echo $list
    # Fetch Medline-formatted files instead of abstract files for future processing
    #fetchquery="efetch.fcgi?db=pubmed&id=$list&rettype=pubmed&retmode=text"
    fetchquery="efetch.fcgi?db=pubmed&id=$list&rettype=medline&retmode=text"
    echo $list
    if [ $DL == "wget" ]; then
        wget ${baseurl}${fetchquery} -O ${currentdate}_fetch.txt
    elif [ $DL =="curl" ]; then
        curl ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
    else
        fetch ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
    fi
    bzip2 ${currentdate}_fetch.txt
fi
