##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import time

import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions

import numpy as np
from sklearn import linear_model, datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"
__credits__ = ["Jasper Snoek", "Ryan P. Adams", "Hugo Larochelle"]

     
def uai_2(params, **kwargs):
    print 'Params: ', params, '\n'
    #y = benchmark_functions.save_svm_on_grid(params, opt_time=ret_time, **kwargs)
    logreg = linear_model.LogisticRegression(penalty=params['penalty'],tol=float(params['tol']),C=float(params['strength']))
    if params['n_min'] > params['n_max']:
      z=params['n_min']
      params['n_min']=params['n_max']
      params['n_max']=z 
    if params['stop_words']==True:
      st='english'
    else:
      st=None
    vectorizer = TfidfVectorizer(ngram_range=(int(params['n_min']),int(params['n_max'])),binary=params['binary'],use_idf=params['idf'],smooth_idf=True, stop_words=st)
    #print 'preprocess data'
    #EDITED FILEPATH FROM DANI'S VERSION:
    data_dir = '/home/jesse/projects/data/trees_binary/'
    ifile=open(data_dir + 'train.data')
    train=[]
    ntrain=0
    for line in ifile:
      train.append(line.strip())
      ntrain=ntrain+1
    ifile.close()
    dev=[]
    ndev=0
    ifile=open(data_dir + 'dev.data')
    for line in ifile:
      train.append(line.strip())
      ndev=ndev+1
    ifile.close()
    ifile=open(data_dir + 'test.data')
    for line in ifile:
      train.append(line.strip())
    ifile.close()
    ifile=open(data_dir + 'train.labels')
    train_label=[]
    for line in ifile:
      train_label.append(line.strip()) 
    ifile.close()
    ifile=open(data_dir + 'test.labels')
    test_label=[]
    for line in ifile:
      test_label.append(line.strip())
    ifile.close()
    ifile=open(data_dir + 'dev.labels')
    dev_label=[]
    for line in ifile:
      dev_label.append(line.strip())
    ifile.close()
    vectors_all = vectorizer.fit_transform(train)
    nrow=ntrain
    vectors=vectors_all[0:nrow,:]
    vectors_dev=vectors_all[nrow:nrow+ndev,:]
    vectors_test=vectors_all[nrow+ndev:,:]
    #print vectors.shape
    #print vectors_test.shape
    print 'fit model'
    logreg.fit(vectors,train_label)
    print 'predict model'
    pred=logreg.predict(vectors_test)
    preddev=logreg.predict(vectors_dev)
    print 'evaluate'
    y=metrics.accuracy_score(dev_label, preddev)
    acc=metrics.accuracy_score(test_label, pred)
    print 'Result: ', y
    print 'Test result: ',acc
    return [-y,acc]


if __name__ == "__main__":
    #args, cli_params = benchmark_util.parse_cli()
    #result = main(cli_params, ret_time=False, **args)
    #duration = main(cli_params, ret_time=True, **args)
    #print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
    #    ("SAT", abs(duration), result, -1, str(__file__))
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    params['penalty']='l2'
    params['tol']=0.098#0.0341863931454#0.0150987537966
    params['strength']=10#20.0092406164#141.096317833
    params['n_min']=1
    params['n_max']=2
    params['binary']=True#False
    params['idf']=True
    params['stop_words']=False
    [result,test_error] = uai_2(params, **args)
    duration = time.time() - starttime
    #params['test_result']=test_error
    print "Result for ParamILS: %s, %f, 1, %f, %f, %d, %s" % \
        ("SAT", abs(duration), result, test_error, -1, str(__file__))
