import json
from optparse import OptionParser
import pandas as pd
import cPickle as pickle
import numpy as np
import os


def main():
    # Handle input options and arguments
    usage = "%prog review_phones_and_codes.pkl output_dir [text.json]"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    input_filename = args[0]
    output_dir = args[1]
    output_filename = os.path.join(output_dir, 'labels.csv')
    if len(args) > 2:
        textfile = args[2]
        with open(textfile, 'r') as input_file:
            text = json.load(input_file)
    else:
        text = None

    with open(input_filename, 'rb') as input_file:
        df = pickle.load(input_file)


    n, p = df.shape
    print df.head()

    #df2 = pd.DataFrame(index=np.arange(n*2), columns=df.columns[1:])

    #df2.ix[:n-1, 1:] = df.ix[:, 2:].as_matrix()
    #df2.ix[n:, 1:] = df.ix[:, 2:].as_matrix()
    #temp = [p[0] for p in df.ix[:, 1].values]
    #df2.ix[:n-1, 0] = temp
    #temp = [p[1] if len(p) > 1 else '' for p in df.ix[:, 1].values]
    #df2.ix[n:, 0] = temp
    """
    df2 = pd.DataFrame(index=df.index, columns=df.columns[1:])
    df2.ix[:, 1:] = np.array(df.ix[:, 2:], dtype=int)
    temp = [p[0] for p in df.ix[:, 1].values]
    df2.ix[:, 0] = temp


    print df2.shape
    indices = df2[df2.phone == ''].index
    print len(indices)
    df2 = df2.drop(df2.index[indices])
    print df2.shape

    print df2.head()

    for c in df2.columns[3:]:
        for v in df2[c].values:
            print v
        print c, set(df2[c].values), np.sum(np.isnan(df2[c].values))
    """

    print df.columns
    df2 = pd.DataFrame(columns=df.columns[2:])

    j = 0
    rows = []
    if text is not None:
        keys = frozenset(text.keys())
        print len(keys)
        for i in df.index:
            phones = df.loc[i, 'phone']
            for phone in phones:
                if phone in keys:
                    #rows.append(pd.DataFrame())
                    df2.loc[phone] = df.loc[i].values[2:].tolist()
                    j += 1
                    if j % 1000 == 0:
                        print j

    #df2 = pd.concat(rows)
    #df2 = pd.DataFrame(df2, columns=df.columns[1:])

    print "saving"
    with open(output_filename, 'wb') as output_file:
        df2.to_csv(output_file)

    for c in df2.columns[1:]:
        df3 = df2[c]
        output_filename = os.path.join(output_dir, c + '.csv')
        df3.to_csv(output_filename)
        print c, df3.sum()


if __name__ == '__main__':
    main()
