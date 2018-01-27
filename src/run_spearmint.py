import spearmint
import space_manager
import spearmint_compatibility.get_spearmint_space
import numpy as np
import importlib
from ExperimentGrid import GridMap
import copy
import hyperopt

def spearmint_main(num_models, model_types, search_space, max_iter, args, call_experiment, model_dir):
    options = set_default_options(args)
    # convert search space to [0,1]^d hypercube
    variables, hparams_to_return_template, transformer = spearmint_compatibility.get_spearmint_space.main(num_models, model_types, search_space)

    # init the chooser
    module  = importlib.import_module('chooser.' + options['chooser_module'], package='spearmint')
    chooser = module.init(model_dir, options['chooser_args'])
    gmap = GridMap(variables, options['grid_size'])

    results = make_result_dict()
    for i in range(max_iter):
        #  get batch of points to evaluate (in [0,1]^d)
        for j in range(options['batch_size']):
            add_next_point_to_pending(chooser, options, variables, results, gmap)

        
        for j in range(len(results['pending'])):
            cur_hparams_to_eval = make_hparams_to_return(results['pending'][j], copy.deepcopy(hparams_to_return_template), gmap, variables, transformer)
            cur_hparams_result = call_experiment(cur_hparams_to_eval)
            
            
        


    #  for assignment in batch:
    #   convert from [0,1]^d to hyperparameters
    #   call_experiment(assignment)
    # return final object which has all times and accuracies stored


def set_default_options(args):
    options = {}
    options['chooser_module'] = 'GPEIOptChooser'
    options['chooser_args'] = ''
    options['grid_size'] = 1000
    options['grid_seed'] = 1

    assert args['algorithm'] in ['spearmint_seq']
    if args['algorithm'] == 'spearmint_seq':
        options['batch_size'] = 2
    
    return options

def make_result_dict():
    results = {}
    results['values'] = np.array([])
    results['complete'] = np.array([])
    results['pending'] = np.array([])
    results['durations'] = np.array([])
    return results                                    

def add_next_point_to_pending(chooser, options, variables, results, gmap):
    

    # Now lets get the next job to run
    # First throw out a set of candidates on the unit hypercube
    # Increment by the number of observed so we don't take the
    # same values twice
    offset = results['pending'].shape[0] + results['complete'].shape[0]
    candidates = gmap.hypercube_grid(options['grid_size'], options['grid_seed']+offset)

    # Ask the chooser to actually pick one.
    # First mash the data into a format that matches that of the other
    # spearmint drivers to pass to the chooser modules.
    grid = candidates
    if (results['complete'].shape[0] > 0):
        grid = np.vstack((results['complete'], candidates))
    if (results['pending'].shape[0] > 0):
        grid = np.vstack((grid, results['pending']))
    grid = np.asarray(grid)
    grid_idx = np.hstack((np.zeros(results['complete'].shape[0]),
                          np.ones(candidates.shape[0]),
                          1.+np.ones(results['pending'].shape[0])))
    job_id = chooser.next(grid, np.squeeze(results['values']), results['durations'],
                          np.nonzero(grid_idx == 1)[0],
                          np.nonzero(grid_idx == 2)[0],
                          np.nonzero(grid_idx == 0)[0])
                          
    # If the job_id is a tuple, then the chooser picked a new job not from
    # the candidate list
    if isinstance(job_id, tuple):
        (job_id, candidate) = job_id
    else:
        candidate = grid[job_id,:]

    if results['pending'].shape[0] > 0:
        results['pending'] = np.vstack((results['pending'], candidate))
    else:
        results['pending'] = np.matrix(candidate)


    print candidate
    params = gmap.unit_to_list(candidate)
    return params
                                    
def make_hparams_to_return(cur_point, hparams_to_return, gmap, variables, transformer):

    cur_point_correct_shape = np.squeeze(np.array(cur_point))
    cur_hparams = gmap.unit_to_list(cur_point_correct_shape)
    for i in range(len(variables)):
        hparams_to_return[variables[i]['name']] = [cur_hparams[i]]

    for k in hparams_to_return.keys():
        hparams_to_return[k] = hparams_to_return[k][0]

    
    # this is gross, but necessary to get the right format, including 'features' and other variables we aren't searching over
    memo = transformer.memo_from_config(hparams_to_return)
    final_to_return = hyperopt.pyll.rec_eval(transformer.expr, memo=memo, print_node_on_error=False)


    return final_to_return
    

