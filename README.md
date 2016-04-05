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
* xgboost

most of the above can be installed using pip. If you have sudo access, run:

    sudo pip install -U numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn

If you don't, and want to install locally, run:

    pip install -U --user numpy scipy pandas pymongo networkx hyperopt nltk scikit-learn

to download the NLKT files, run 

    python -m nltk.downloader all

for additional info on nltk (e.g. if you'd like to install its data into another location) look here: http://www.nltk.org/data.html


installing XGBoost (on linux):

#move to a directory where you can install xgboost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package
python setup.py install