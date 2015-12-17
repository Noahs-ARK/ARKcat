

import sys

#files to convert:
dir_name = sys.argv[1]

types_of_data = ['train','dev','test']
for t in types_of_data:

    ifile = open(dir_name + '/' + t + '.data', 'r')
    ofile = open(dir_name + '/' + t + '.json', 'w')
    ofile.write('{\n')
    
    lines = ifile.readlines()
    for index, line in enumerate(lines):
        index = index + 1
        if index == len(lines): 
            ofile.write('  "' + str(index) + '": "' + line[:-2].replace('\*','*') + '"\n')
        else:
            ofile.write('  "' + str(index) + '": "' + line[:-2].replace('\*','*') + '",\n')
                 
    ofile.write('}')
    ofile.close()
    ifile.close()



    ifile = open(dir_name + '/' + t + '.labels', 'r')
    ofile = open(dir_name + '/' + t + '.csv', 'w')
    ofile.write('idontknow,whattoputhere\n')
    
    lines = ifile.readlines()
    for index, line in enumerate(lines):
        index = index + 1
        ofile.write(str(index) + ',' + line)
    ofile.close()
    ifile.close()
