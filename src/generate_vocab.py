import sys, re
from sklearn.feature_extraction.text import TfidfVectorizer
from cnn_methods import *
import argparse
# import scipy

def main(files):
    word2vec_file_path = files[0]
    output_file_path = files[1]
    input_file_paths = files[2:]
    print 'processing files:', input_file_paths, 'using word vector file', word2vec_file_path
    print 'outputting to', output_file_path
    vocab = []
    line_counter = 0

    vectorizer = TfidfVectorizer(input='filename')
    vectorizer.fit(input_file_paths)
    for word in vectorizer.vocabulary_:
        word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", word)
        if not (word in vocab):
            vocab.append(word.encode('ascii', 'replace'))

    print "len vocab =", len(vocab)
    with open(output_file_path, 'w') as output_file:
        with open(word2vec_file_path) as word2vec:
            while True:
                line = word2vec.readline()
                if not line:
                    break
                else:
                    tokens = tokenize(line)
                word, vector = tokens[0], tokens[1:]
                word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", word)
                if word in vocab:
                    output_file.write(word + ' ')
                    for token in vector:
                        output_file.write(token + ' ')
                    output_file.write('\n')
                    del vocab[vocab.index(word)]
                    line_counter += 1
    print 'len file =', line_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='need to write one')
    parser.add_argument('files', nargs='+', help='')
    main(vars(parser.parse_args(sys.argv[1:]))['files'])
