import re
import json
import gzip
import codecs
import pandas as pd
import cPickle as pickle
from optparse import OptionParser

def find_phones(filename, phones):
    with gzip.open(filename, 'rb' ) as input_file:
        for line in input_file:
            temp = line.decode('utf-8', 'replace')
            parts = temp.split('\t')
            if len(parts) > 3:
                text = parts[3]
                try:
                    metadata = json.loads(parts[4], encoding='utf-8')
                    phone = metadata.get(u'phone', None)
                    if type(phone) == unicode or type(phone) == str:
                        phone = re.sub('-', '', phone)
                        if phone[0] == '1':
                            phone = phone[1:]
                        if phone in phones:
                            yield (phone, text)
                    elif type(phone) == list:
                        print "list of numbers", phone
                except ValueError, e:
                    #print e.message, parts[4]
                    pass



def main():
    """ test function """
    # Handle input options and arguments
    usage = "%prog ads.tsv.gz reviews_phones_and_features.pkl output_filename"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()
    ads_filename = args[0]
    labels_filename = args[1]
    output_filename = args[2]

    with open(labels_filename, 'rb') as input_file:
        df = pickle.load(input_file)

    print df.ix[:5, 0].values
    phones = frozenset(df.ix[:, 0].values)

    data = {k: v for (k, v) in find_phones(ads_filename, phones)}

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=2)


if __name__ == '__main__':
    main()
