import os
import ast
from collections import Counter
from scipy import sparse
import random

import numpy as np

import vocabulary_with_counts
import file_handling as fh

class FeatureExtractorCounts:

    dirname = None
    name = None
    prefix = None
    n = None
    min_doc_threshold = None
    binarize = None

    index = None
    vocab = None
    column_names = None

    def __init__(self, basedir, name, prefix, min_doc_threshold=1, binarize=True):
        self.name = name
        self.prefix = prefix
        self.min_doc_threshold = int(min_doc_threshold)
        self.binarize = ast.literal_eval(str(binarize))
        self.feature_counts = None
        self.index = None
        self.vocab = None
        self.make_dirname(basedir)

    def get_name(self):
        return self.name

    def get_prefix(self):
        return self.prefix

    def get_min_doc_threshold(self):
        return self.min_doc_threshold

    def get_binarize(self):
        return self.binarize

    def get_dirname(self):
        return self.dirname

    def make_dirname(self, basedir):
        #dirname = ','.join([self.name, 'mdt=' + str(self.min_doc_threshold), 'bin=' + str(self.binarize)])
        self.dirname = os.path.join(basedir, self.name)

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

        # don't prune now; do it as a transformation
        #vocab.prune(min_doc_threshold=self.get_min_doc_threshold())

        #if verbose:
        #    print "Vocabulary size after pruning:", len(vocab)

        return vocab

    def extract_feature_counts(self, items, tokens, vocab):
        n_items = len(items)
        n_features = len(vocab)

        row_starts_and_ends = [0]
        column_indices = []
        values = []
        #oov_counts = []

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
            #if self.get_binarize():
            #    values.extend([1]*len(token_counts))
            #else:
            #    values.extend(token_counts)

            #oov_counts.append(token_counter.get(vocab.oov_index, 0))
            row_starts_and_ends.append(len(column_indices))

        dtype = 'int32'

        feature_counts = sparse.csr_matrix((values, column_indices, row_starts_and_ends), dtype=dtype)

        assert feature_counts.shape[0] == n_items
        assert feature_counts.shape[1] == n_features

        return feature_counts

    def load_vocabulary(self):
        vocab = vocabulary_with_counts.VocabWithCounts(self.get_prefix(), add_oov=True,
                                                       read_from_filename=self.get_vocab_filename())
        return vocab

    def load_from_files(self):
        self.vocab = self.load_vocabulary()

        index = fh.read_json(self.get_index_filename())
        feature_counts = fh.unpickle_data(self.get_feature_filename())
        #oov_counts = fh.read_json(self.get_oov_count_filename())

        self.feature_counts = feature_counts
        self.index = index
        self.column_names = np.array(self.vocab.index2token)
        #self.oov_counts = oov_counts
        self.do_transformations()

    def do_transformations(self):
        """Apply thresholding and binarization"""

        # threshold by min_doc_threshold
        temp = self.feature_counts.copy().tolil()
        n, p = temp.shape

        orig_vocab_index = self.vocab.index2token[:]

        if self.min_doc_threshold > 1:
            print "Thresholding"

            # prune the vocabulary
            self.vocab.prune(self.min_doc_threshold)

            # convert this to an index into the columns of the feature matrix
            index = np.array([k for (k, v) in enumerate(orig_vocab_index) if v in self.vocab.index2token])

            # make sure we include the OOV column:


            #feature_sums = np.array(temp.sum(axis=0))
            #feature_sums = np.reshape(feature_sums, p)
            # select the first colum (OOV) automatically, and then all the rest
            #feature_sums[0] = self.min_doc_threshold
            #feature_sel = np.array(feature_sums >= self.min_doc_threshold)
            #index = np.arange(p)[feature_sel]

            print "Size before thresholding", temp.shape
            thresholded = temp[:, index]
            print "Size after thresholding", thresholded.shape

            # add counts of out-of-vocabulary words based on this thresholding
            #neg_index = np.arange(p)[np.array(1-feature_sel, dtype=bool)]
            #oov_counts = temp[:, neg_index]
            #oov_sums = oov_counts.sum(axis=1)
            #thresholded[:, 0] = oov_sums

            self.feature_counts = thresholded.copy().tocsr()
            self.column_names = self.vocab.index2token

        # binarize counts
        if self.binarize:
            print "Binarizing"
            self.feature_counts = sparse.csr_matrix(self.feature_counts > 0, dtype=int)

    def get_counts(self):
        return self.index, self.column_names, self.feature_counts

    def get_vocab_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'vocab', 'json')

    def get_index_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'index', 'json')
