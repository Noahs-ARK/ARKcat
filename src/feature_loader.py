import os
import sys
from optparse import OptionParser

import numpy as np

from feature_extractor_ngrams import FeatureExtractorCountsNgrams
from feature_extractor_chargrams import FeatureExtractorCountsCharGrams

def load_feature(feature_description, feature_dir, source, items_to_load, vocab_source=None, verbose=1):
    """Load the feature counts associated with a given feature for a list of items

    Args:
        feature_description (string): starts with feature type, followed by options (comma-separated no spaces)
        feature_dir (string): directory where extracted features will be written
        source (string): json filename with all document text
        verbose (int): level of verbosity
        vocab_source (string): equivalent of feature_dir when loading a vocab extracted from training data

    Returns:
        np.array: vector of item labels (n,)
        np.array: vector of column names (c,)
        np.array: array of labels (n, c)
    """
    parts = feature_description.split(',')
    feature_name = parts[0]
    extractor = None
    if feature_name == 'ngrams':
        extractor = extractor_factory(FeatureExtractorCountsNgrams, feature_dir, kwargs_list_to_dict(parts[1:]))
    elif feature_name == 'chargrams':
        extractor = extractor_factory(FeatureExtractorCountsCharGrams, feature_dir, kwargs_list_to_dict(parts[1:]))
    else:
        sys.exit("Feature name " + feature_name + " not recognized")

    print extractor.get_dirname()
    if not os.path.exists(extractor.get_dirname()):
        if verbose > 0:
            print "Extracting", feature_description
        extractor.extract_features(source, write_to_file=True, vocab_source=vocab_source)
    else:
        if verbose > 1:
            print "Loading", extractor.get_full_name()
        extractor.load_from_files()

    index, column_names, counts = extractor.get_counts()

    indices_to_load = [index[i] for i in items_to_load]
    return items_to_load, np.array(column_names), counts[indices_to_load, :]


def kwargs_list_to_dict(list_of_kwargs):
    kwargs = {}
    for kwarg in list_of_kwargs:
        name, value = kwarg.split('=')
        if value[0] == '[' and value[-1] == ']':
            parts = value[1:-1].split(';')
            value = parts
        kwargs[name] = value
    return kwargs


def extractor_factory(extractor_class, feature_dir, kwargs):
    return extractor_class(feature_dir, **kwargs)



def main():
    """ test function """
    # Handle input options and arguments
    usage = "%prog text.json feature_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-f', dest='feature', default='ngrams',
                      help='Feature type to load; default=%default')
    parser.add_option('-m', dest='min_doc_thresh', default=1,
                      help='Minimum document threshold; default=%default')
    parser.add_option('-b', dest='binarize', action="store_true", default=False,
                      help='Binarize counts; default=%default')
    parser.add_option('-n', dest='ngrams', default=1,
                      help='n for ngrams; default=%default')

    (options, args) = parser.parse_args()
    n = int(options.ngrams)
    min_doc_thresh = int(options.min_doc_thresh)
    binarize = options.binarize
    feature_type = options.feature

    data_file = args[0]
    feature_dir = args[1]

    feature_description = ','.join([feature_type, 'min_df=' + str(min_doc_thresh),
                                    'binarize=' + str(binarize), 'n=' + str(n)])

    items, columns, data = load_feature(feature_description, feature_dir, data_file)
    print items[:5]
    print columns[:5]
    print data[:5, :5]

if __name__ == '__main__':
    main()
