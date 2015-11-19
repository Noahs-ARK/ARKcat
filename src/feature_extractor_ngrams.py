import numpy as np

import tokenizer
import file_handling as fh
from feature_extractor import FeatureExtractorCounts


class FeatureExtractorCountsNgrams(FeatureExtractorCounts):

    n = None

    def __init__(self, basedir, n=1, min_doc_threshold=1, binarize=True):
        name = 'ngrams'
        self.n = int(n)
        prefix = '_n' + str(n) + '_'
        FeatureExtractorCounts.__init__(self, basedir, 'ngrams', prefix, min_doc_threshold=min_doc_threshold,
                                        binarize=binarize)
        self.extend_dirname()

    def extend_dirname(self):
        self.dirname += ',n=' + str(self.n)

    def get_n(self):
        return self.n

    def extract_features(self, source, write_to_file=True):
        print "Extracting ngram tokens"

        # read in a dict of {document_key: text}
        data = fh.read_json(source)
        all_items = data.keys()

        tokens = self.extract_tokens_from_file(data, self.get_n())

        vocab = self.make_vocabulary(tokens, all_items)

        #feature_counts, oov_counts = self.extract_feature_counts(all_items, tokens, vocab)
        feature_counts = self.extract_feature_counts(all_items, tokens, vocab)

        if write_to_file:
            vocab.write_to_file(self.get_vocab_filename())
            fh.write_to_json(all_items, self.get_index_filename(), sort_keys=False)
            fh.pickle_data(feature_counts, self.get_feature_filename())
            #fh.write_to_json(oov_counts, self.get_oov_count_filename(), sort_keys=False)

        self.feature_counts = feature_counts
        self.index = all_items
        self.vocab = vocab
        self.column_names = np.array(self.vocab.index2token)
        self.do_transformations()

    def extract_tokens_from_file(self, data, n):
        token_dict = {}
        for key, text in data.items():
            text = text.lower()
            text = text.lstrip()
            text = text.rstrip()
            tokens = []

            sentences = tokenizer.split_sentences(text)
            for s in sentences:
                sent_tokens = tokenizer.make_ngrams(s, n)
                tokens = tokens + sent_tokens

            tokens = [self.get_prefix() + t for t in tokens]
            token_dict[key] = tokens
        return token_dict

    def do_transformations(self):
        FeatureExtractorCounts.do_transformations(self)





