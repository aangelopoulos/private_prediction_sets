mkdir ./cifar10/.cache
mkdir ./cifar10/data
mkdir ./cifar10/models
mkdir ./cifar10/outputs
mkdir ./cifar10/outputs/histograms
mkdir ./imagenet/.cache
mkdir ./imagenet/outputs
mkdir ./imagenet/outputs/histograms
mkdir ./covid-chest-xray
mkdir ./covid-chest-xray/outputs
mkdir ./covid-chest-xray/outputs/histograms
mkdir ./covid-chest-xray/outputs/three-covid
mkdir ./scripts/outputs
mkdir ./scripts/.cache
mkdir ./core/.cache
conda env create -f environment.yml
conda activate pps 
wget -O ./covid-chest-xray/data.zip https://berkeley.box.com/shared/static/53zbuo93lq55wxxk9jcigufwp61wh4gm.zip -q --show-progress
unzip covid-chest-xray/data.zip -d covid-chest-xray
rm covid-chest-xray/data.zip
wget -O ./cifar10/models.zip https://berkeley.box.com/shared/static/rrkgdlnkgqwru2cnob4ebvx1yvod5rfe.zip -q --show-progress
unzip cifar10/models.zip -d cifar10
rm cifar10/models.zip
