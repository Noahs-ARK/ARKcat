import spearmint
import space_manager
import spearmint_compatibility.get_spearmint_space
    
    

def spearmint_main(num_models, model_types, search_space, max_iter, args, call_experiment):
    # convert search space to [0,1]^d hypercube
    # this is a hyperopt.unif_hparam_sample.Unif_Sampler object, relevant fields: dists, index_names
    uniform_sampler = spearmint_compatibility.get_spearmint_space.main(num_models, model_types, search_space)
    
    for i in range(max_iter):
        #  get batch of points to evaluate (in [0,1]^d)
        
        
    #  convert them to hyperparameters
    #  for assignment in batch:
    #   convert from [0,1]^d to hyperparameters
    #   call_experiment(assignment)
    # return final object which has all times and accuracies stored




