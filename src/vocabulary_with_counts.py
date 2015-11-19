from collections import Counter

from vocabulary import Vocab
import file_handling as fh

class VocabWithCounts(Vocab):

    counts = None
    doc_counts = None

    def __init__(self, prefix, add_oov=True, read_from_filename=None, tokens_to_add=None):
        Vocab.__init__(self, prefix, add_oov=add_oov)
        self.counts = Counter()
        self.doc_counts = Counter()

        if read_from_filename is not None:
            self.read_from_file(read_from_filename)

        if tokens_to_add is not None:
            self.add_tokens(tokens_to_add)

    def add_tokens(self, tokens):
        Vocab.add_tokens(self, tokens)

        if self.counts is not None:
            self.counts.update(tokens)

        if self.doc_counts is not None:
            token_set = set(tokens)
            self.doc_counts.update(token_set)

    def get_count(self, token):
        return self.counts.get(token, 0)

    def get_counts(self, tokens):
        counts = [self.get_count(t) for t in tokens]
        return counts

    def get_count_from_index(self, index):
        token = self.get_token(index)
        return self.get_count(token)

    def get_counts_from_indices(self, indices):
        tokens = self.get_tokens(indices)
        return self.get_counts(tokens)

    def prune(self, min_doc_threshold=1):
        self.index2token = [t for t in self.index2token if self.doc_counts[t] >= min_doc_threshold]
        if self.oov_token not in self.index2token and self.oov_index > -1:
            self.index2token.insert(self.oov_index, self.oov_token)
        self.token2index = dict(zip(self.index2token, range(len(self.token2index))))
        self.counts = Counter({t: c for (t, c) in self.counts.items() if t in self.token2index})
        self.doc_counts = Counter({t: c for (t, c) in self.doc_counts.items() if t in self.token2index})

    def write_to_file(self, filename):
        json_obj = {'index2token': self.index2token, 'counts': self.counts, 'doc_counts': self.doc_counts}
        fh.write_to_json(json_obj, filename, sort_keys=False)

    def read_from_file(self, filename):
        json_obj = fh.read_json(filename)
        self.index2token = json_obj['index2token']
        self.counts = Counter(json_obj['counts'])
        self.doc_counts = Counter(json_obj['doc_counts'])
        self.token2index = dict(zip(self.index2token, range(len(self.index2token))))
