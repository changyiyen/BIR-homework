## Version 2022-08-02
Set-Variable -Name baseurl -Value "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" -Description "Base URL for Eutils queries."
Set-Variable -Name searchquery_base -Value "esearch.fcgi?db=pubmed&" -Description "Search using esearch."
# Modify search terms as needed
Set-Variable -Name searchquery_term -Value "term=HLA+Antigens[Mesh]+AND+Lupus+Erythematosus+,+Systemic[Mesh]" -Description "Search terms."
# Modify search parameters as needed (default: sort by publication date, most recent 20 entries only)
Set-Variable -Name searchquery_param -Value "&sort=pubdate&retmax=20" -Description "Search parameters."
Set-Variable -Name searchquery -Value ${searchquery_base}${searchquery_term}${searchquery_param} -Description "Actual search query."
Set-Variable -Name currentdate -Value (Get-Date -Format "yyyy-MM-dd") -Description "Today's date in ISO8601."
Invoke-WebRequest -Uri ((Get-Variable baseurl -value) + (Get-Variable searchquery -value)) -OutFile ((Get-Variable currentdate -value)+"_search.xml")
$xmldoc = [xml](Get-Content ((Get-Variable currentdate -value)+"_search.xml"))
Set-Variable -Name pmid -Value (Select-Xml -Xml ${xmldoc} -XPath "eSearchResult/IdList/Id" | foreach {$_.node.InnerXml})
$pmid = $pmid -join ','
Set-Variable -Name fetchquery -Value ("efetch.fcgi?db=pubmed&id="+$pmid+"&rettype=medline&retmode=text")
Invoke-WebRequest -Uri ${baseurl}${fetchquery} -OutFile ${currentdate}_fetch.txt
# Using zip here since bzip2 isn't available by default on Windows
Compress-Archive -Path ${currentdate}_fetch.txt -DestinationPath ${currentdate}_fetch.zip