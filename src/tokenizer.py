import re

import nltk.data
from nltk.stem import WordNetLemmatizer


def split_sentences(text, decorate=False):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.sentences_from_text(text, realign_boundaries=True)
    if decorate:
        sentences = [sent + ' <SE>' for sent in sentences]
    return sentences


def split_into_words(text, lemmatize=False, reattach=True, replace_numbers=True, split_off_quotes=True,
                     fix_semicolon_mistakes=True):

    if fix_semicolon_mistakes:
        text = fix_semicolons(text)

    word_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    # get rid of certain character so that we can use those for special purposes
    tokens = word_tokenizer.tokenize(text)
    if reattach:
        tokens = reattach_clitics(tokens)

    if split_off_quotes:
        tokens = split_off_quote_marks(tokens)

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if replace_numbers:
        pattern = '^[0-9]+$'
        tokens = [t if re.search(pattern, t) is None else '__NUM__' for t in tokens]

    tokens = split_tokens(tokens, '.,')

    return tokens


def reattach_clitics(tokens):
    #clitic_pattern = '^\'(s|S|d|D|ve|VE|t|T|m|M|re|RE|ll|LL)'
    #clitic_pattern = "^(n't|'ll|'re|'ve)"

    clitic_pattern = "^((n't)|('s)|('m)|('re)|('ve)|('ll)|('d)|('l)|('t))$"

    pop_list = []
    # append clitics to previous words
    for i in range(1, len(tokens)):
        if re.search(clitic_pattern, tokens[i]):
            tokens[i-1] += tokens[i]
            if i not in pop_list:
                pop_list.append(i)

    pop_list.sort()
    pop_list.reverse()
    for i in pop_list:
        tokens.pop(i)

    return tokens

def split_off_quote_marks(tokens):
    i = 0
    pattern1 = r"^('+)(.+)"
    while i < len(tokens):
        token = tokens[i]
        match = re.search(pattern1, token)
        if match is not None:

            tokens[i] = match.group(1)
            tokens.insert(i+1, match.group(2))
        i += 1
    return tokens


def fix_semicolons(text):
    pattern = "([a-z]+;(t|s|m))[^a-z]"
    match = re.search(pattern, text)
    if match is not None:
        repl = re.sub(';', "'", match.group(1))
        text = re.sub(match.group(1), repl, text)
    return text

def make_ngrams(text, n, lemmatize=False, reattach=True, replace_numbers=True, split_off_quotes=True):
    tokens = split_into_words(text, lemmatize=lemmatize, reattach=reattach, replace_numbers=replace_numbers,
                              split_off_quotes=split_off_quotes)
    if n > 1:
        N = len(tokens)
        grams = [tokens[k:N-(n-1-k)] for k in range(n)]
        tokens = map(u'_'.join, zip(*grams))
    return tokens


def split_tokens(tokens, delimiters):
    # split on and keep periods
    tokens = [re.split('([' + delimiters + '])', token) for token in tokens]
    # flatten
    tokens = [token for sublist in tokens for token in sublist]
    tokens = [token for token in tokens if token != '']
    return tokens

