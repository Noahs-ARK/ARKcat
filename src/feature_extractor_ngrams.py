import numpy as np

import tokenizer
import file_handling as fh
from feature_extractor import FeatureExtractorCounts


class FeatureExtractorCountsNgrams(FeatureExtractorCounts):

    n = None

    def __init__(self, basedir, n=1, min_doc_threshold=1, transform=None):
        name = 'ngrams'
        self.n = int(n)
        prefix = '_n' + str(n) + '_'
        FeatureExtractorCounts.__init__(self, basedir, name, prefix, min_doc_threshold=min_doc_threshold,
                                        transform=transform)
        self.extend_dirname()

    def extend_dirname(self):
        self.dirname += ',n=' + str(self.n)

    def get_n(self):
        return self.n

    def extract_tokens_from_text(self, data):
        token_dict = {}
        for key, text in data.items():
            text = text.lower()
            text = text.lstrip()
            text = text.rstrip()
            tokens = []

            sentences = tokenizer.split_sentences(text)
            for s in sentences:
                sent_tokens = tokenizer.make_ngrams(s, self.n)
                tokens = tokens + sent_tokens

            tokens = [self.get_prefix() + t for t in tokens]
            token_dict[key] = tokens
        return token_dict

    def do_transformations(self):
        FeatureExtractorCounts.do_transformations(self)





