import random
from random_search import *
from grid_search import *
import os
import sys
import argparse

# def main(search_type, search_space, model_type='cnn', num_models=None):
def main(args):
    if args['grid']:
        grid = get_grid([args['model_type']], args['search_space'][0])
        num_models = len(grid.grid)
        search_type = 'grid'
    elif args['rand']:
        num_models = args['rand'][0]
        search_type = 'rand'
        grid = print_random_search(num_models, [args['model_type']], args['search_space'][0])
    else:
        print 'Search type not defined'
        raise UnboundLocalError
    file_path = os.path.expanduser('~') + '/datasets/' + search_type + '_'
    file_path += args['search_space'][0] + '_' + args['model_type'] + '_' + str(num_models) + '.models'
    with open(file_path, 'w+') as outfile:
        if args['rand']:
            outfile.write(print_grid(grid))
        else:
            outfile.write(print_grid(grid.grid))

#randomly chooses hyperparameters from given grid
def print_random_search(num_models, model_type, search_space):
    random.seed(None)
    list_of_models = []
    for model in range(num_models):
        if model_type[0] == 'cnn':
            model_dict = {'model_0': get_cnn_model(str('0'), search_space),
                          'features_0': {'nmin_to_max_0': (1,1),
                                  'binary_0': False,
                                  'use_idf_0': False,
                                  'st_wrd_0': None}}
            list_of_models.append(model_dict)
        # elif model_type == 'linear':
        #     list_of_models.append(str('0'))
        else:
            raise NotImplementedError
    random.shuffle(list_of_models)
    return list_of_models

def print_grid(grid):
    random.seed(None)
    random.shuffle(grid)
    string = ''
    for element in grid:
        string += str(element) + '\n'
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='need to write one')
    parser.add_argument('search_space', nargs=1, type=str,
            choices=['reg', 'arch', 'big', 'small'], help='specify search space')
    parser.add_argument('model_type', nargs='?', type=str, default='cnn', choices=['cnn', 'linear'])
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-g', '--grid', dest='grid', action='store_true', help='')
    parser.add_argument('-r', '--rand', nargs=1, dest='rand', type=int,
                help='random search; specify iterations with optional addl argument; default=%default')
    print sys.argv
    print vars(parser.parse_args(sys.argv[1:]))
    main(vars(parser.parse_args(sys.argv[1:])))
