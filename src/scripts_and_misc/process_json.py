# directory = 'Congressional_Bill_Corpus.v1.00/raw/'
directory = ''
text_file = directory + 'billtext_org.json'
labels_file = directory + 'train.json'
output_dir = '/Users/katya/datasets/congress_bills_2/'

import sys

#pcogennen noah's congress_bills_2 into useable format 

#to dict:
def skip_ahead_n_quotes(line, char_counter, maximum):
        quote_counter = 0
        while quote_counter < maximum:
            if line[char_counter:char_counter+1] == '\"':
                quote_counter += 1
            char_counter += 1
        # print 'to',line[char_counter:char_counter+10]
        return char_counter

def parse_inside_char(line, char_counter, char):
        string = ''
        while line[char_counter] != char:
            string += line[char_counter]
            char_counter += 1
        return string, char_counter

def rm_newlines(string):
    # string.replace('\\\n', ' ')
    string = string.replace('\\' + 'n', ' ')
    for i in range(1,10):
        string = string.replace('  ', ' ')
    return string

def tokenize(line):
   list_of_words = []
   word = ''
   for char in line:
      if char == ' ':
         list_of_words.append(word)
         word = ''
      else:
         word += char
   list_of_words.append(word.strip())
   return tuple(list_of_words)

d = {}
for line in open(text_file):
    if "\"\"" in line:
        d[name] = ''
    else:
        # d = json.load(json_data)
        # print d
        char_counter = 0
        # print "success"
        name, char_counter = parse_inside_char(line, char_counter, '\t')
        # print 'parse'
        if '\"body\"' in line:
            char_counter = skip_ahead_n_quotes(line, char_counter, 2)
            # print 'skip ahead'
            char_counter += 3
            body, char_counter = parse_inside_char(line, char_counter, '\"')
            # print 'parsed'
        else:
            body = ''
        char_counter = skip_ahead_n_quotes(line, char_counter, 3)
        char_counter += 3
        # print 'skip 2'
        title, char_counter = parse_inside_char(line, char_counter, '\"')
        # print 'parsed2'
        d[name] = rm_newlines(title) + ' ' + rm_newlines(body)
print 'quit'
with open(labels_file, 'r') as labels, open(output_dir + 'train.data', 'w') as data_out, open(output_dir + 'train.labels', 'w') as labels_out:
    for line in labels:
                line = line.replace('\t', ' ')
                example_name, label = tokenize(line)
                try:
                    data_out.write(d[example_name] + '\n')
                except KeyError:
                    print example_name
                else:
                    labels_out.write(label + '\n')
                sys.stdout.flush()
print 'done'
