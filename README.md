##Dependencies
The following dependencies are required:

* hyperopt
* networkx
* pymongo
* nltk
* scikit-learn
* pandas
* numpy
* scipy

all of the above can be installed using pip. If you have sudo access, run:

    sudo pip install -U numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn

If you don't, and want to install locally, run:

    pip install -U --user numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn

to download the NLKT files, run 

    python -m nltk.downloader all

for additional info on nltk (e.g. if you'd like to install its data into another location) look here: http://www.nltk.org/data.html
