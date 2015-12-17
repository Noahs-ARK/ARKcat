import os
import ast
from collections import Counter
from scipy import sparse

import numpy as np

import vocabulary_with_counts
import file_handling as fh

class FeatureExtractorCounts:

    dirname = None
    name = None
    prefix = None
    n = None
    min_df = None
    binarize = None

    index = None
    vocab = None
    column_names = None

    def __init__(self, basedir, name, prefix, min_df=1, transform=None):
        self.name = name
        self.prefix = prefix
        self.min_df = int(min_df)
        self.transform = transform
        self.feature_counts = None
        self.index = None
        self.vocab = None
        self.make_dirname(basedir)

    def get_name(self):
        return self.name

    def get_prefix(self):
        return self.prefix

    def get_min_df(self):
        return self.min_df

    def get_transform(self):
        return self.transform

    def get_dirname(self):
        return self.dirname

    def make_dirname(self, basedir):
        self.dirname = os.path.join(basedir, self.name + ',min_df=' + str(self.min_df))

    def get_feature_filename(self):
        return fh.make_filename(fh.makedirs(self.dirname), 'counts', 'pkl')

    def get_oov_count_filename(self):
        return fh.make_filename(fh.makedirs(self.dirname), 'oov_counts', 'json')

    def make_vocabulary(self, tokens, items, verbose=True):
        if verbose:
            print "Making vocabulary for", self.get_name()

        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True)

        for item in items:
            vocab.add_tokens(tokens[item])

        return vocab

    def extract_features(self, source, write_to_file=True, vocab_source=None):
        print "Extracting ngram tokens"

        # read in a dict of {document_key: text}
        data = fh.read_json(source)
        all_items = data.keys()

        tokens = self.extract_tokens_from_text(data)

        if vocab_source is None:
            vocab = self.make_vocabulary(tokens, all_items)
            vocab.prune(self.min_df)
            self.vocab = vocab
        else:
            vocab = self.load_vocabulary(vocab_source)
            self.vocab = vocab

        feature_counts, index = self.extract_feature_counts(all_items, tokens, vocab)

        if write_to_file:
            vocab.write_to_file(self.get_vocab_filename())
            fh.write_to_json(index, self.get_index_filename(), sort_keys=False)
            fh.pickle_data(feature_counts, self.get_feature_filename())

        self.feature_counts = feature_counts
        self.index = index
        self.column_names = np.array(self.vocab.index2token)
        self.do_transformations()

    def extract_tokens_from_text(self, data):
        """
        Basic way of converting feature input into lists of tokens.
        Override this in your feature extractor
        :param data: dictionary of key-value pairs, where value could be text (as here), a list, or whatever
        :return: token_dict: dictionary of key:value pairs, where the values are lists of tokens to be counted
        """
        token_dict = {}
        for key, text in data.items():
            text = text.lower()
            text = text.lstrip()
            text = text.rstrip()
            tokens = text.split()
            token_dict[key] = tokens
        return token_dict

    def extract_feature_counts(self, items, tokens, vocab):
        n_items = len(items)
        n_features = len(vocab)

        row_starts_and_ends = [0]
        column_indices = []
        values = []

        for item in items:
            # get the index for each token
            token_indices = vocab.get_indices(tokens[item])

            # count how many times each index appears
            token_counter = Counter(token_indices)
            token_keys = token_counter.keys()
            token_counts = token_counter.values()

            # put it into the from of a sparse matix
            column_indices.extend(token_keys)
            values.extend(token_counts)
            row_starts_and_ends.append(len(column_indices))

        dtype = 'float'

        feature_counts = sparse.csr_matrix((values, column_indices, row_starts_and_ends),
                                           shape=(n_items, n_features), dtype=dtype)

        #print max(column_indices)
        print feature_counts.shape[0] == n_items
        print feature_counts.shape[1] == n_features

        index = dict(zip(items, range(len(items))))
        return feature_counts, index

    def load_vocabulary(self, vocab_source=None):
        if vocab_source is None:
            vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True,
                                                           read_from_filename=self.get_vocab_filename())
        else:
            vocab_filename = os.path.join(vocab_source, os.path.basename(self.dirname), 'vocab.json')
            vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True,
                                                           read_from_filename=vocab_filename)
        return vocab

    def load_from_files(self, vocab_source=None):
        self.vocab = self.load_vocabulary(vocab_source=vocab_source)

        index = fh.read_json(self.get_index_filename())
        feature_counts = fh.unpickle_data(self.get_feature_filename())

        self.feature_counts = feature_counts
        self.index = index
        self.column_names = np.array(self.vocab.index2token)
        self.do_transformations()

    def do_transformations(self):
        """Apply thresholding and transforms"""

        if self.transform == 'binarize':
            print "Binarizing"
            self.feature_counts = sparse.csr_matrix(self.feature_counts > 0, dtype=int)

        elif self.transform == 'tfidf':
            print "Doing tf-idf transform"
            doc_sums = self.feature_counts.sum(axis=1)
            tf = sparse.csr_matrix(self.feature_counts.multiply(1.0/doc_sums))
            doc_counts = self.vocab.get_all_doc_counts()
            n_docs = doc_counts.max()
            idf = np.log(float(n_docs) / doc_counts)
            self.feature_counts = sparse.csr_matrix(tf.multiply(idf))

    def get_counts(self):
        return self.index, self.column_names, self.feature_counts

    def get_vocab_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'vocab', 'json')

    def get_index_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'index', 'json')
