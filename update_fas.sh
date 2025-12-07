#!/opt/homebrew/bin/fish

rsynct requirements.txt $fas/clusterattention
rsynct transformer $fas/clusterattention
rsynct variants $fas/clusterattention
rsynct runs $fas/clusterattention
rsynct train.py $fas/clusterattention
rsynct enwik8 $fas/clusterattention
rsynct *.sh $fas/clusterattention
rsynct *.py $fas/clusterattention
rsynct data $fas/clusterattention