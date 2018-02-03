import spearmint
import space_manager
import spearmint_compatibility.get_spearmint_space
import numpy as np
import importlib
from ExperimentGrid import GridMap
import copy
import hyperopt
import Queue
import time
from hyperopt.unif_hparam_sample import Unif_Sampler

def spearmint_main(num_models, model_types, search_space, max_iter, args, call_experiment, model_dir):
    options = set_default_options(args, max_iter)
    # convert search space to [0,1]^d hypercube
    variables, hparams_to_return_template, transformer = spearmint_compatibility.get_spearmint_space.main(num_models,
                                                                                                          model_types, search_space)

    # init the chooser
    module  = importlib.import_module('chooser.' + options['chooser_module'], package='spearmint')
    chooser = module.init(model_dir, options['chooser_args'])
    gmap = GridMap(variables, options['grid_size'])

    results = make_result_dict()
    for i in range(options['num_iters']):
        #  get batch of points to evaluate (in [0,1]^d)
        for j in range(options['batch_size']):
            add_next_point_to_pending(chooser, options, variables, results, gmap)
            

        # to evaluate the batch of points selected
        while len(results['pending']) > 0:
            cur_hparams_to_eval = make_hparams_to_return(results['pending'][0], copy.deepcopy(hparams_to_return_template),
                                                         gmap, variables, transformer)                
            cur_hparams_result = call_experiment(cur_hparams_to_eval)
            update_results(cur_hparams_result, results, cur_hparams_to_eval)

    
    printing_best_results(results, variables)
    return results


def set_default_options(args, max_iter):
    
    options = {}
    options['chooser_module'] = 'GPEIOptChooser'
    options['chooser_args'] = ''
    options['grid_size'] = 1000
    options['grid_seed'] = 1

    assert args['algorithm'] == 'spearmint_seq' or args['algorithm'] == 'spearmint_kover2' or args['algorithm'] == 'sobol_noise', "invalid search algo"
    
    if args['algorithm'] == 'spearmint_seq':
        options['batch_size'] = 1
        options['num_iters'] = max_iter
    elif args['algorithm'] == 'spearmint_kover2':
        assert max_iter % 2 == 0
        options['batch_size'] = int(max_iter / 2)
        options['num_iters'] = 2
    elif args['algorithm'] == 'sobol_noise':
        options['batch_size'] = max_iter
        options['num_iters'] = 1
    
        
    return options

def make_result_dict():
    results = {}
    results['values'] = np.array([])
    results['complete'] = np.array([])
    results['pending'] = np.array([])
    results['durations'] = np.array([])
    results['durations_spearmint'] = []
    return results                                    

def add_next_point_to_pending(chooser, options, variables, results, gmap):
    
    start_time = time.time()
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

    try:
        if results['values'].shape[0] == 0:
            candidate += np.random.random_sample(candidate.shape)
            candidate -= np.floor(candidate)
    except:
        pass

    # apparently spearmint will occasionally return points *outside* of the [0,1]^d hypercube
    # due to machine precision errors. this causes problems.
    for i in range(len(candidate)):
        assert candidate[i] > -0.001 and candidate[i] < 1.001
        if candidate[i] < 0:
            candidate[i] = 0
        if candidate[i] > 1:
            candidate[i] = 1

    
    if results['pending'].shape[0] > 0:
        results['pending'] = np.vstack((results['pending'], candidate))
    else:
        results['pending'] = np.matrix(candidate)


    results['durations_spearmint'].append(time.time() - start_time)
                                    
def make_hparams_to_return(cur_point, hparams_to_return, gmap, variables, transformer):

    
    cur_point_correct_shape = np.squeeze(np.array(cur_point))
    cur_hparams = gmap.unit_to_list(cur_point_correct_shape)
    for i in range(len(variables)):
        hparams_to_return[variables[i]['name']] = [cur_hparams[i]]

    for k in hparams_to_return.keys():
        hparams_to_return[k] = hparams_to_return[k][0]                

    # to get the nodes which need to be transformed into logspace:
    us = Unif_Sampler(transformer.expr)
    for i in range(len(us.index_names)):
        if isinstance(us.dists[i], hyperopt.hparam_distribution_sampler.LogUniform):
            hparams_to_return[us.index_names[i]] = np.exp(hparams_to_return[us.index_names[i]])
            
        
    
    # this is gross, but necessary to get the right format, including 'features' and other variables we aren't searching over
    memo = transformer.memo_from_config(hparams_to_return)
    final_to_return = hyperopt.pyll.rec_eval(transformer.expr, memo=memo, print_node_on_error=False)

    return final_to_return
    

def update_results(cur_results, results, cur_hparams_evaluated):
    #import pdb; pdb.set_trace()

    val = cur_results['loss']
    dur = cur_results['duration']
    variables = results['pending'][0]
    
    # to update the completed section
    if results['complete'].shape[0] > 0:
        results['values'] = np.vstack((results['values'], float(val)))
        results['complete'] = np.vstack((results['complete'], variables))
        results['durations'] = np.vstack((results['durations'], float(dur)))
        results['hparams_evaluated'].append(cur_hparams_evaluated)
    else:
        results['values'] = float(val)
        results['complete'] = variables
        results['durations'] = float(dur)
        results['hparams_evaluated'] = [cur_hparams_evaluated]

    # to update the 'pending' section by removing the no-longer-pending assignment
    results['pending'] = np.delete(results['pending'], 0, axis=0)
    
    
def printing_best_results(results, variables):
    print("all hparams tried:")
    #print results['complete']
    for i in range(len(results['complete'])):
        print i, results['complete'][i]
    print("durations for training each model:")
    print results['durations'].tolist()
    print("the amount of time it took for spearmint to select the next example:")
    print results['durations_spearmint']
    print("variables:")
    for var in variables:
        print var['name']
    print("all hparams (in hparam space):")
    for i in range(len(results['hparams_evaluated'])):
        print i, results['hparams_evaluated'][i]

    print ("number of points evaluated: {}".format(len(results['hparams_evaluated'])))

    priority_q = Queue.PriorityQueue()
    if isinstance(results['values'], float):
        losses = [results['values']]
    else:
        losses = results['values']
    for i in range(len(losses)):
        priority_q.put((losses[i],i))
    print('top losses and settings: ')
    for i in range(0, min(3, len(losses))):
        index = priority_q.get()[1]
        print(losses[index])
        print(results['hparams_evaluated'][index])
        print("")
