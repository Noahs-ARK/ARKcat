from hyperopt import fmin, tpe, hp, Trials, space_eval, rand, anneal
from hyperopt import dpp, dpp_random, sample_hparam_space
import Queue as queue
import space_manager



def printing_best(trials):
    priority_q = queue.PriorityQueue()
    losses = trials.losses()
    for i in range(len(losses)):
        priority_q.put((losses[i], i))
    print('top losses and settings: ')
    for i in range(0,min(3,max_iter)):
        index = priority_q.get()[1]
        print(losses[index])
        print(trials.trials[index]['misc']['vals'])
        print('')
    print('')


def set_discretize_num(trials, search_space):
    if search_space == 'arch':
        trials.discretize_num = 4
    elif 'reg' in search_space:
        trials.discretize_num = 15
    elif 'debug' in search_space:
        trials.discretize_num = None
    elif 'default' in search_space:
        trials.discretize_num = 5
    else:
        raise ValueError("you tried to use " + search_space + " as a search space, but we don't know how many "+
                         "values we should discretize to (for the dpp)")


def hyperopt_main(num_models, model_types, search_space, max_iter, args, call_experiment):

    trials = Trials()
    trials.discretize_space = True

    # a hacky solution to pass parameters to hyperopt
    if trials.discretize_space:
        set_discretize_num(trials, search_space)

    if args['run_bayesopt']:
        space = space_manager.get_space(num_models, model_types, search_space)
        if args['algorithm'] == "bayes_opt":
            algorithm = tpe.suggest
        elif args['algorithm'] == "random":
            algorithm = rand.suggest
        elif args['algorithm'] == "anneal":
            algorithm = anneal.suggest
        elif args['algorithm'] == "dpp_cos":
            trials.dpp_dist = "cos"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_ham":
            trials.dpp_dist = "ham"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_l2":
            trials.dpp_dist = "l2"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_rbf":
            trials.dpp_dist = "rbf"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_rbf_narrow":
            trials.dpp_dist = "rbf_narrow"
            algorithm = dpp.suggest
        elif args['algorithm'] == 'mixed_dpp_rbf':
            trials.dpp_dist = "rbf"
            algorithm = dpp.suggest
            trials.discretize_space = False
        elif args['algorithm'] == 'mixed_dpp_rbf_clip':
            trials.dpp_dist = "rbf_clip"
            algorithm = dpp.suggest
            trials.discretize_space = False
        elif args['algorithm'] == 'mixed_dpp_rbf_narrow':
            trials.dpp_dist = "rbf_narrow"
            algorithm = dpp.suggest
            trials.discretize_space = False
        elif args['algorithm'] == 'mixed_dpp_rbf_vnarrow':
            trials.dpp_dist = "rbf_vnarrow"
            algorithm = dpp.suggest
            trials.discretize_space = False
        elif args['algorithm'] == "dpp_random":
            algorithm = dpp_random.suggest
        else:
            raise NameError("Unknown algorithm for search")

        #DEBUGGING: this is for profiling. it prints where the program has spent the most time
        #profile = cProfile.Profile()
        import pdb; pdb.set_trace()
        #tmp = sample_hparam_space(space, algorithm, max_iter, 'l2', True, 15)
        try:
            #profile.enable()

            best = fmin(call_experiment,
                        space=space,
                        algo=algorithm,
                        max_evals=max_iter,
                        trials=trials)
            #profile.disable()
        finally:
            #profile = pstats.Stats(profile).sort_stats('cumulative')
            #profile.print_stats()
            print('')

        print space_eval(space, best)
        printing_best(trials)

    #loading models from file
    else:
        with open(model_path) as f:
            for i in range(line_num - 1):
                f.readline()

            space = eval(f.readline())
            best = call_experiment(space)
    print("the total runtime: {}".format(time.time() - start_time))

