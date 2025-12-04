#!/opt/homebrew/bin/fish

rsynct requirements.txt $fas/clusterattention
rsynct transformer $fas/clusterattention
rsynct variants $fas/clusterattention
rsynct runs $fas/clusterattention
rsynct train.py $fas/clusterattention
rsynct enwik8 $fas/clusterattention
rsynct run.sh $fas/clusterattention
rsynct run_hep.sh $fas/clusterattention
rsynct train_hep.py $fas/clusterattention
rsynct train_hep_v2.py $fas/clusterattention
rsynct train_modelnet.py $fas/clusterattention
rsynct data $fas/clusterattention