from sklearn.datasets import fetch_20newsgroups
import os, sys
import subprocess, random
import unicodedata




base = "/cab1/corpora/bayes_opt/20_newsgroups/"


groups = {'all_topics':['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism'], 
          'science':['sci.space', 'sci.crypt', 'sci.med', 'sci.electronics'], 
          'religion':['talk.religion.misc', 'alt.atheism'],
          'comp':['comp.windows.x', 'comp.graphics']}            
random.seed(999)





def do_stuff(train_or_test):
    make_dev = train_or_test == 'train'
    for group in groups:
        #make directory
        os.system('mkdir -p ' + base + group)
        
        examples = {}
        newsgroups_data = fetch_20newsgroups(subset=train_or_test, categories=groups[group], remove=('headers'))
        for i in range(len(newsgroups_data['data'])):
            line = newsgroups_data['data'][i]
#            line = unicode(line, errors='ignore')
            unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')
            line = ''.join(ch for ch in line if unicodedata.category(ch)[0]!="C")
            contents = line.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\0',' ').replace('*', '.').strip().replace('"', "''").replace("\\","\\\\")

            examples[len(examples)] = (contents, newsgroups_data['target_names'][newsgroups_data['target'][i]])
        
        
        

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
                    dev_json_out.write('  "' + str(dev_counter) + '": "' + examples[index][0].replace('\*','*').encode('ascii', 'ignore') + '"\n')
                else:
                    dev_json_out.write('  "' + str(dev_counter) + '": "' + examples[index][0].replace('\*','*').encode('ascii', 'ignore') + '",\n')
                dev_csv_out.write(str(dev_counter) + ',' + str(examples[index][1]) + '\n')
            else:
                train_counter = train_counter + 1
                train_json_out.write('  "' + str(train_counter) + '": "' + examples[index][0].replace('\*','*').encode('ascii', 'ignore') + '",\n')
                train_csv_out.write(str(train_counter) + ',' + str(examples[index][1]) + '\n')

        train_json_out.write('}')
        if make_dev:
            dev_json_out.write('}')
        print("DON'T FORGET TO REMOVE THE EXTRA COMMA AT THE END OF TRAIN.JSON AND TEST.JSON")

        train_json_out.close()
        train_csv_out.close()
        if make_dev:
            dev_json_out.close()
            dev_csv_out.close()


do_stuff('train')
do_stuff('test')
#dev_items = sorted(random.sample(xrange(len(examples)), int(.2*len(examples))))
