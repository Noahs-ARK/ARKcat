import os, sys
import subprocess, random
import unicodedata

"""
20 Newsgroups (Lang, 1995): the 20
Newsgroups dataset is a benchmark topic
classification dataset, we use the publicly
available copy at http://qwone.com/
jason/20Newsgroups. There are 20 topics
in this dataset. We derived four topic
classification tasks from this dataset. The
first task is to classify documents across all
20 topics. The second task is to classify
related science documents into four science
topics (sci.crypt, sci.electronics,
sci.med, sci.med). 3 The third and
fourth tasks are talk.religion.misc
vs. alt.atheism and comp.graphics
vs. comp.windows.x. To consider a more
realistic setting, we removed header information
from each article since they often contain label
information.
"""



base = "/cab1/corpora/bayes_opt/20_newsgroups/"
original_train = base + 'original/20news-bydate-train/'
original_test = base + 'original/20news-bydate-test/'


groups = {'all_topics':['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism'], 
          'science':['sci.space', 'sci.crypt', 'sci.med', 'sci.electronics'], 
          'religion':['talk.religion.misc', 'alt.atheism'],
          'comp':['comp.windows.x', 'comp.graphics']}            
            

    
#sys.exit(0)


def read_one_file(f_in):
    contents = ''
    f = open(f_in, 'r')
    lines = f.readlines()
    
    first_header_removed = False
    second_header_removed = False
    for line in lines:
        if line == '\n':
            if not first_header_removed:
                first_header_removed = True
            elif not second_header_removed:
                second_header_removed = True
        if second_header_removed:
            
            line = unicode(line, errors='ignore')
            unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')
            line = ''.join(ch for ch in line if unicodedata.category(ch)[0]!="C")
            contents = contents + line.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\0',' ').replace('*', '.')#.replace('\\\*','*').replace('\\*','*').replace('\*','*')

    return contents.strip().replace('"', "''").replace("\\","\\\\")


random.seed(999)


def do_stuff(original_loc, make_dev):
    for group in groups:
        #make directory
        os.system('mkdir -p ' + base + group)
        examples = {}
        for topic in groups[group]:
            for f in os.listdir(original_loc + topic):
                lines = read_one_file(original_loc + topic + '/' + f)

                examples[len(examples)] = (lines, topic)

        dev_items = sorted(random.sample(xrange(len(examples)), int(.2*len(examples))))
        if not make_dev:
            dev_items = []
        print(len(dev_items))
        
        if make_dev:
            train_json_out = open(base + group + '/train.json', 'w')
            train_csv_out = open(base + group + '/train.csv', 'w')
            dev_json_out = open(base + group + '/dev.json', 'w')
            dev_csv_out = open(base + group + '/dev.csv', 'w')
        else:
            train_json_out = open(base + group + '/test.json', 'w')
            train_csv_out = open(base + group + '/test.csv', 'w')
            
        
        train_json_out.write('{\n')
        train_csv_out.write('idontknow,whattoputhere\n')
        if make_dev:
            dev_csv_out.write('idontknow,whattoputhere\n')
            dev_json_out.write('{\n')

        dev_counter = 0
        train_counter = 0
        for index in examples:
            if index in dev_items:
                dev_counter = dev_counter + 1
                if index == dev_items[len(dev_items)-1]:
                    dev_json_out.write('  "' + str(dev_counter) + '": "' + examples[index][0].replace('\*','*') + '"\n')
                else:
                    dev_json_out.write('  "' + str(dev_counter) + '": "' + examples[index][0].replace('\*','*') + '",\n')
                dev_csv_out.write(str(dev_counter) + ',' + examples[index][1] + '\n')
            else:
                train_counter = train_counter + 1
                train_json_out.write('  "' + str(train_counter) + '": "' + examples[index][0].replace('\*','*') + '",\n')
                train_csv_out.write(str(train_counter) + ',' + examples[index][1] + '\n')
    
        train_json_out.write('}')
        if make_dev:
            dev_json_out.write('}')
        print("DON'T FORGET TO REMOVE THE EXTRA COMMA AT THE END OF TRAIN.JSON AND TEST.JSON")
    
        train_json_out.close()
        train_csv_out.close()
        if make_dev:
            dev_json_out.close()
            dev_csv_out.close()
    
    


do_stuff(original_train, True)
do_stuff(original_test, False)
