import os
import sys
from optparse import OptionParser

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

assert os.path.basename(os.getcwd()) == 'src'
sys.path.append(os.getcwd())
import file_handling as fh


def main():

    usage = "%prog [category1 category2 ...]"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    parser.add_option('--remove_headers', action="store_true", dest="remove_headers", default=False,
                      help='Remove headers: default=%default')

    parser.add_option('--remove_footers', action="store_true", dest="remove_footers", default=False,
                      help='Remove footers: default=%default')

    parser.add_option('--remove_quotes', action="store_true", dest="remove_quotes", default=False,
                      help='Remove quotes: default=%default')


    (options, args) = parser.parse_args()
    categories = args[:]
    remove_headers = options.remove_headers
    remove_footers = options.remove_footers
    remove_quotes = options.remove_quotes
    export_20ng(remove_headers, remove_footers, remove_quotes, categories)


def export_20ng(remove_headers=False, remove_footers=False, remove_quotes=False, categories=None):
    output_dir = os.path.join('..', 'datasets', '20ng', 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    remove = []
    if remove_headers:
        remove.append('headers')
    if remove_footers:
        remove.append('footers')
    if remove_quotes:
        remove.append('quotes')

    print categories

    ng_train = fetch_20newsgroups(subset='train', remove=remove, categories=categories)
    keys = ['train' + str(i) for i in range(len(ng_train.data))]
    print len(keys)
    train_text = dict(zip(keys, ng_train.data))
    fh.write_to_json(train_text, os.path.join(output_dir, 'train.json'))

    train_labels = pd.DataFrame(ng_train.target, columns=['target'], index=keys)
    train_labels.to_csv(os.path.join(output_dir, 'train.csv'))
    print train_labels.shape

    ng_test = fetch_20newsgroups(subset='test', remove=remove, categories=categories)
    keys = ['test' + str(i) for i in range(len(ng_test.data))]
    test_text = dict(zip(keys, ng_train.data))
    fh.write_to_json(test_text, os.path.join(output_dir, 'test.json'))

    test_labels = pd.DataFrame(ng_test.target, columns=['target'], index=keys)
    test_labels.to_csv(os.path.join(output_dir, 'test.csv'))




if __name__ == '__main__':
    main()


