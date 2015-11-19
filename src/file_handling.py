import os
import json
import glob
import codecs

import cPickle as pickle

import pandas as pd

def get_basename(input_filename):
    parts = os.path.split(input_filename)
    # deal with the situation in which we're given a directory (ending with a pathsep)
    if parts[1] == '':
        parts = os.path.split(parts[0])
    basename = os.path.splitext(parts[1])[0]
    return basename

def make_filename(base_dir, input_filename, extension):
    basename = get_basename(input_filename)
    filename = os.path.join(makedirs(base_dir), basename + '.' + extension)
    return filename

def makedirs(*args):
    dirname = os.path.join(*args)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def ls(input_dir, query):
    if query != '':
        search_string = os.path.join(input_dir, query)
    else:
        search_string = os.path.join(input_dir, '*')
    files = glob.glob(search_string)
    return files


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)

def read_json(input_filename):
    with codecs.open(input_filename, 'r') as input_file:
        temp = json.load(input_file)
        if isinstance(temp, dict):
            # try convert keys back to ints, if appropriate
            try:
                data = {int(k): v for (k, v) in temp.items()}
            except ValueError, e:
                data = temp
        else:
            data = temp
    return data


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r') as input_file:
        #lines = input_file.read().split('\n')
        lines = input_file.readlines()
    return lines

def read_csv(input_filename):
    return pd.read_csv(input_filename, header=0, index_col=0)

