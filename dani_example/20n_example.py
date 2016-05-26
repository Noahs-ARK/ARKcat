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

#import HPOlib.benchmark_util as benchmark_util
#import HPOlib.benchmark_functions as benchmark_functions
import sys, os
import numpy as np
from sklearn import linear_model, datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"
__credits__ = ["Jasper Snoek", "Ryan P. Adams", "Hugo Larochelle"]

     
def uai(params):#, **kwargs):
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
    vectorizer = TfidfVectorizer(ngram_range=(int(params['n_min']),int(params['n_max'])),binary=params['binary'],use_idf=params['idf'],smooth_idf=True,stop_words=st)
    if params['cats'] == 'all':
        cats = None
    elif params['cats'] == 'science':
        cats = ['sci.med','sci.space','sci.crypt','sci.electronics']
    elif params['cats'] == 'religion':
        cats = ['alt.atheism', 'talk.religion.misc']
    elif params['cats'] == 'graphics':
        cats = ['comp.windows.x','comp.graphics']
    #cats = ['sci.med','sci.space']
    #cats = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
    print 'preprocess data'
    #newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    #vectors = vectorizer.fit_transform(newsgroups_train.data)
    #print vectors.shape
    #newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)   
    #print 'preprocess test data'
    #vectors_test = vectorizer.fit_transform(newsgroups_test.data)
    if params['rm_footers']:
        to_remove = ('headers', 'footers')
    else:
        to_remove = ('headers',)
        
    print_20n(to_remove, cats, params)
    newsgroups_all = fetch_20newsgroups(subset='all', remove=to_remove, categories=cats)#,'footers'))#,'footers','quotes'), categories=cats)   
    vectors_all = vectorizer.fit_transform(newsgroups_all.data)
    #nrow=round(7.0/10.0*vectors_all.shape[0])
    newsgroups_train = fetch_20newsgroups(subset='train',remove=to_remove, categories=cats)
    nrow=newsgroups_train.target.shape[0]
    #print nrow
    #print vectors_all.shape
    vectors=vectors_all[0:nrow,:]
    vectors_test=vectors_all[nrow:,:]
    #print vectors.shape
    #print vectors_test.shape
    print 'fit model'
    logreg.fit(vectors,newsgroups_all.target[0:nrow])
    print 'predict model'
    pred=logreg.predict(vectors_test)
    print 'evaluate'
    y=metrics.accuracy_score(newsgroups_all.target[nrow:], pred)
    print 'Result: ', y
    print('idf: ', params['idf'], 'rm_footers: ', params['rm_footers'], 'cats: ', params['cats'])
    return -y

def print_20n(to_remove, cats, params):
    train = fetch_20newsgroups(subset='train',remove=to_remove, categories=cats)
    test = fetch_20newsgroups(subset='test',remove=to_remove, categories=cats)
    out_loc = "/cab1/corpora/bayes_opt/20_newsgroups_from_danis_code/"
    os.system('mkdir -p ' + out_loc + params['cats'])
    with open(out_loc + params['cats'] + '/train.labels', 'w') as labels:
        with open(out_loc + params['cats'] + '/train.data', 'w') as data_out:
            for i in range(len(train['data'])):
                data_out.write(train['data'][i].replace('\n', ' ').encode('utf8') + '\n')
                labels.write(train['target_names'][train['target'][i]] + '\n')
    with open(out_loc + params['cats'] + '/test.labels', 'w') as labels:
        with open(out_loc + params['cats'] + '/test.data', 'w') as data_out :
            for i in range(len(test['data'])):
                data_out.write(test['data'][i].replace('\n', ' ').encode('utf8') + '\n')
                labels.write(test['target_names'][test['target'][i]] + '\n')
    sys.exit(0)
    


if __name__ == "__main__":
    #args, cli_params = benchmark_util.parse_cli()
    #result = main(cli_params, ret_time=False, **args)
    #duration = main(cli_params, ret_time=True, **args)
    #print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
    #    ("SAT", abs(duration), result, -1, str(__file__))
    starttime = time.time()
    #args, params = benchmark_util.parse_cli()
    print(sys.argv[1])
    params = {}
    params['penalty']=sys.argv[1]
    params['tol']=float(sys.argv[2])#0.098#0.0341863931454#0.0150987537966
    params['strength']=float(sys.argv[3])#20.0092406164#141.096317833
    params['n_min']=int(sys.argv[4])
    params['n_max']=int(sys.argv[5])
    params['binary']= sys.argv[6] == 'True'#True#False
    params['idf']=sys.argv[7] == 'True'
    params['stop_words']=sys.argv[8] == 'True'
    params['cats']=sys.argv[9]
    params['rm_footers']=sys.argv[10] == 'True'
    print params
    result = uai(params)#, **args)
    
