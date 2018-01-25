import space_manager
import hyperopt
from hyperopt.unif_hparam_sample import Unif_Sampler

def main(num_models, model_types, search_space):
    import pdb; pdb.set_trace()
    space = space_manager.get_space(num_models, model_types, search_space)
    us = Unif_Sampler(hyperopt.pyll.as_apply(space))
    us.index_names
    return us
    


    
