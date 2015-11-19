# Adapted from http://nbviewer.ipython.org/github/JonathanRaiman/theano_lstm/blob/master/Tutorial.ipynb

import numpy as np
import file_handling as fh

class Vocab:

    def __init__(self, prefix, read_from_filename=None, tokens_to_add=None, add_oov=True):
        self.token2index = {}
        self.index2token = []

        self.oov_index = -1
        self.oov_token = prefix + '__OOV__'

        if add_oov:
            self.oov_index = 0
            self.add_tokens([self.oov_token])

        if read_from_filename is not None:
            self.read_from_file(read_from_filename)

        if tokens_to_add is not None:
            self.add_tokens(tokens_to_add)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.token2index:
                self.token2index[token] = len(self.token2index)
                self.index2token.append(token)

    def get_token(self, index):
        return self.index2token[index]

    def get_tokens(self, indices):
        tokens = [self.index2token[i] for i in indices]
        return tokens

    def get_all_tokens(self):
        return self.index2token

    def get_index(self, token):
        return self.token2index.get(token, self.oov_index)

    def get_indices(self, tokens):
        indices = np.zeros(len(tokens), dtype=np.int32)
        for i, token in enumerate(tokens):
            indices[i] = self.token2index.get(token, self.oov_index)
        return indices

    @property
    def size(self):
        return len(self.index2token)

    def __len__(self):
        return len(self.index2token)

    def sort(self):
        self.index2token.sort()
        self.token2index = dict(zip(self.index2token, range(len(self.token2index))))

    def write_to_file(self, filename):
        fh.write_to_json(self.index2token, filename, sort_keys=False)

    def read_from_file(self, filename):
        self.index2token = fh.read_json(filename)
        self.token2index = dict(zip(self.index2token, range(len(self.index2token))))

