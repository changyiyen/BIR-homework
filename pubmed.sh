#!/usr/env sh
# initial version: 2022-08-08
# current version: 2023-10-12
# dependencies: XMLStarlet, (wget or curl)

# NB. Some behavior here only works on Bash (if on GNU/Linux) and doesn't work on other shells (e.g. dash) 

usage() {
    printf "Usage: $0 [-h][-m][-s <n>]\n" 1&>2;
	printf "       -h: displays this help\n" 1&>2;
	printf "       -m: downloads the PubMed XML file for each entry (multiple files)\n" 1&>2;
    printf "       -s <n>: sleep for a random period, up to 'n' seconds\n" 1&>2;
	exit 1; 
}

## Change this to whereever you would like the results placed
#DLDIR="/home/cyyen/ncbi-pubmed"
DLDIR="./als_corpus"
echo "[INFO] Download directory set to $DLDIR."

if [ ! -e "$DLDIR" ]; then
    echo "[INFO] Creating download directory $DLDIR."
	mkdir $DLDIR
fi

cd $DLDIR
echo "[INFO] Changed working directory to $DLDIR."

while getopts ":hms:" option; do
    case "${option}" in
        m)
            m=true ;;
		s)
		    s=${OPTARG} ;;
        *)
            usage ;;
    esac
done
shift $((OPTIND-1))

if [ ! -e "`which xmlstarlet`" ]; then
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
if [ -n "$s" ]; then
    SLEEPTIME=0
    if [ `uname -s` = 'FreeBSD' ]; then
        SLEEPTIME=`jot -r 1 1 $s`
    elif [ `uname -s` = 'Linux' ]; then
        SLEEPTIME=$((RANDOM % $s))
    fi
	echo "[INFO] Sleeping for $SLEEPTIME seconds."
    sleep $SLEEPTIME
fi

baseurl="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Set PubMed as database to search
searchquery_base="esearch.fcgi?db=pubmed&"

## Search query terms; modify these as needed ##
searchquery_term="term=Amyotrophic+Lateral+Sclerosis[Mesh]"
#searchquery_term="term=Wounds+,+Gunshot[Mesh]"
searchquery_param="&sort=pubdate&retmax=5000"
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
list=`xmlstarlet fo -D ${currentdate}_search.xml | xmlstarlet sel -t -v "eSearchResult/IdList/Id"`
if [ "$m" = true ]; then
    for i in $list; do
        fetchquery="efetch.fcgi?db=pubmed&id=$i&rettype=pubmed&retmode=text"
        if [ $DL == "wget" ]; then
            wget ${baseurl}${fetchquery} -O ${i}.xml
        elif [ $DL =="curl" ]; then
            curl ${baseurl}${fetchquery} -o ${i}.xml
        else
            fetch ${baseurl}${fetchquery} -o ${i}.xml
        fi
    done
else
    list=`xmlstarlet fo -D ${currentdate}_search.xml | xmlstarlet sel -t -v "eSearchResult/IdList/Id" | tr '\n' ','`
    # Fetch Medline-formatted entries into a single bzipped file for future processing
    #fetchquery="efetch.fcgi?db=pubmed&id=$list&rettype=pubmed&retmode=text"
    fetchquery="efetch.fcgi?db=pubmed&id=$list&rettype=medline&retmode=text"
    if [ $DL == "wget" ]; then
        wget ${baseurl}${fetchquery} -O ${currentdate}_fetch.txt
    elif [ $DL =="curl" ]; then
        curl ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
    else
        fetch ${baseurl}${fetchquery} -o ${currentdate}_fetch.txt
    fi
    bzip2 ${currentdate}_fetch.txt
fi
